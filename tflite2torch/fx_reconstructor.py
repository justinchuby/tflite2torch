"""
TFLite execution graph reconstruction in Torch FX.

This module reconstructs the TFLite computational graph as a PyTorch FX graph
and creates a torch.export.ExportedProgram.
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import Argument

from .parser import SubgraphInfo, TensorInfo, OperatorInfo
from .operator_converter import OperatorConverter


class FXReconstructor:
    """
    Reconstructs TFLite execution graph as PyTorch FX graph.

    This class takes parsed TFLite graph information and converts it
    into a PyTorch FX graph that can be executed or exported.
    """

    def __init__(self, operator_converter: Optional[OperatorConverter] = None):
        self.operator_converter = operator_converter or OperatorConverter()
        self.graph = Graph()
        self.tensor_map: Dict[int, Node] = {}
        self.parameter_dict: Dict[str, torch.Tensor] = {}
        self.node_counter = 0

    def reconstruct(self, subgraph: SubgraphInfo, weights: Optional[Dict[int, torch.Tensor]] = None) -> GraphModule:
        """
        Reconstruct a TFLite subgraph as a PyTorch FX GraphModule.

        Args:
            subgraph: Parsed TFLite subgraph information
            weights: Optional dictionary mapping tensor indices to weight tensors

        Returns:
            PyTorch FX GraphModule representing the computation
        """
        weights = weights or {}
        self.graph = Graph()
        self.tensor_map = {}
        self.parameter_dict = {}
        self.node_counter = 0

        # Create placeholder nodes for inputs
        for input_idx in subgraph.inputs:
            tensor_info = subgraph.tensors[input_idx]
            placeholder = self.graph.placeholder(
                name=self._sanitize_name(tensor_info.name or f"input_{input_idx}")
            )
            self.tensor_map[input_idx] = placeholder

        # Process each operator in the subgraph
        for op_idx, operator in enumerate(subgraph.operators):
            self._process_operator(operator, subgraph, weights, op_idx)

        # Create output nodes
        output_nodes = [self.tensor_map[idx] for idx in subgraph.outputs]
        if len(output_nodes) == 1:
            self.graph.output(output_nodes[0])
        else:
            self.graph.output(tuple(output_nodes))

        # Create a module to hold the graph
        root_module = self._create_root_module()
        graph_module = GraphModule(root_module, self.graph)

        return graph_module

    def _process_operator(
        self,
        operator: OperatorInfo,
        subgraph: SubgraphInfo,
        weights: Dict[int, torch.Tensor],
        op_idx: int,
    ):
        """Process a single operator and add it to the FX graph."""
        op_type = operator.op_type
        
        # Get input nodes
        input_nodes = []
        for input_idx in operator.inputs:
            if input_idx in self.tensor_map:
                input_nodes.append(self.tensor_map[input_idx])
            elif input_idx in weights:
                # Create a get_attr node for weights
                tensor_info = subgraph.tensors[input_idx]
                param_name = self._sanitize_name(
                    tensor_info.name or f"param_{input_idx}"
                )
                self.parameter_dict[param_name] = weights[input_idx]
                param_node = self.graph.get_attr(param_name)
                self.tensor_map[input_idx] = param_node
                input_nodes.append(param_node)
            else:
                # Create a placeholder for missing weights/constants
                tensor_info = subgraph.tensors[input_idx]
                param_name = self._sanitize_name(
                    tensor_info.name or f"const_{input_idx}"
                )
                # Create dummy tensor based on shape
                shape = tensor_info.shape
                dummy_tensor = torch.zeros(shape)
                self.parameter_dict[param_name] = dummy_tensor
                param_node = self.graph.get_attr(param_name)
                self.tensor_map[input_idx] = param_node
                input_nodes.append(param_node)

        # Convert operator
        try:
            conv_info = self.operator_converter.convert(
                op_type, operator.inputs, operator.builtin_options
            )
        except NotImplementedError:
            # Create a placeholder for unsupported operators
            node_name = f"unsupported_{op_type.lower()}_{op_idx}"
            output_node = self.graph.call_function(
                lambda *args: args[0] if args else None,
                args=tuple(input_nodes) if input_nodes else ()
            )
            output_node.name = node_name
            for output_idx in operator.outputs:
                self.tensor_map[output_idx] = output_node
            return

        # Create FX node based on conversion info
        node_name = f"{op_type.lower()}_{op_idx}"
        
        if isinstance(conv_info["module"], type) and issubclass(
            conv_info["module"], nn.Module
        ):
            # It's a module class - create a module instance
            # Infer shape parameters from tensors if needed
            params = conv_info["params"].copy()
            
            # For Conv2d, extract in_channels, out_channels, kernel_size from weight tensor
            if conv_info["module"] == nn.Conv2d and len(operator.inputs) >= 2:
                weight_idx = operator.inputs[1]
                weight_tensor_info = subgraph.tensors[weight_idx]
                # TFLite weight format: [out_channels, kernel_h, kernel_w, in_channels]
                if len(weight_tensor_info.shape) == 4:
                    params["out_channels"] = weight_tensor_info.shape[0]
                    params["kernel_size"] = (weight_tensor_info.shape[1], weight_tensor_info.shape[2])
                    params["in_channels"] = weight_tensor_info.shape[3]
                    
            # For Linear, extract in_features, out_features from weight tensor
            elif conv_info["module"] == nn.Linear and len(operator.inputs) >= 2:
                weight_idx = operator.inputs[1]
                weight_tensor_info = subgraph.tensors[weight_idx]
                if len(weight_tensor_info.shape) == 2:
                    params["out_features"] = weight_tensor_info.shape[0]
                    params["in_features"] = weight_tensor_info.shape[1]
            
            module = conv_info["module"](**params)
            module_name = f"module_{self.node_counter}"
            self.node_counter += 1
            self.parameter_dict[module_name] = module
            
            output_node = self.graph.call_module(
                module_name,
                args=tuple(input_nodes)
            )
            output_node.name = node_name
        else:
            # It's a function
            output_node = self.graph.call_function(
                conv_info["module"],
                args=tuple(input_nodes),
                kwargs=conv_info["params"]
            )
            output_node.name = node_name

        # Handle fused activation
        if "activation" in conv_info and conv_info["activation"] != "NONE":
            activation_module = self.operator_converter.get_activation_module(
                conv_info["activation"]
            )
            if activation_module is not None:
                act_name = f"activation_{self.node_counter}"
                self.node_counter += 1
                self.parameter_dict[act_name] = activation_module
                output_node = self.graph.call_module(
                    act_name,
                    args=(output_node,)
                )
                output_node.name = f"{node_name}_activation"

        # Map output tensor indices to the output node
        for output_idx in operator.outputs:
            self.tensor_map[output_idx] = output_node

    def _create_root_module(self) -> nn.Module:
        """Create a module containing all parameters and sub-modules."""
        root = nn.Module()
        for name, value in self.parameter_dict.items():
            if isinstance(value, nn.Module):
                root.add_module(name, value)
            elif isinstance(value, torch.Tensor):
                root.register_parameter(name, nn.Parameter(value))
        return root

    def _sanitize_name(self, name: str) -> str:
        """Sanitize tensor/parameter names for use in FX graph."""
        # Replace invalid characters
        name = name.replace("/", "_")
        name = name.replace(":", "_")
        name = name.replace("-", "_")
        name = name.replace(".", "_")
        
        # Ensure it starts with a letter or underscore
        if name and not (name[0].isalpha() or name[0] == "_"):
            name = f"_{name}"
        
        return name or "unnamed"

    def to_exported_program(self, graph_module: GraphModule) -> Optional[Any]:
        """
        Convert GraphModule to torch.export.ExportedProgram if available.

        Args:
            graph_module: FX GraphModule to export

        Returns:
            ExportedProgram if torch.export is available, otherwise None
        """
        try:
            # Try to use torch.export if available (PyTorch 2.0+)
            if hasattr(torch, "export"):
                # Create example inputs based on placeholders
                example_inputs = []
                for node in graph_module.graph.nodes:
                    if node.op == "placeholder":
                        # Create a dummy input tensor
                        # In production, this should use actual input shapes
                        example_inputs.append(torch.randn(1, 3, 224, 224))
                
                if example_inputs:
                    exported_program = torch.export.export(
                        graph_module,
                        tuple(example_inputs)
                    )
                    return exported_program
        except (AttributeError, Exception) as e:
            # torch.export might not be available or export might fail
            print(f"Warning: Could not create ExportedProgram: {e}")
        
        return None

    def visualize_graph(self, graph_module: GraphModule) -> str:
        """
        Generate a text representation of the FX graph.

        Args:
            graph_module: FX GraphModule to visualize

        Returns:
            String representation of the graph
        """
        lines = []
        lines.append("FX Graph:")
        lines.append("=" * 50)
        
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                lines.append(f"{node.name}: placeholder")
            elif node.op == "get_attr":
                lines.append(f"{node.name}: get_attr({node.target})")
            elif node.op == "call_function":
                args_str = ", ".join(str(arg.name) if isinstance(arg, Node) else str(arg) for arg in node.args)
                lines.append(f"{node.name}: {node.target.__name__}({args_str})")
            elif node.op == "call_module":
                args_str = ", ".join(str(arg.name) if isinstance(arg, Node) else str(arg) for arg in node.args)
                lines.append(f"{node.name}: {node.target}({args_str})")
            elif node.op == "output":
                lines.append(f"output: {node.args}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
