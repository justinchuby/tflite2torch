"""
TFLite execution graph reconstruction in Torch FX.

This module reconstructs the TFLite computational graph as a PyTorch FX graph
and creates a torch.export.ExportedProgram.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._parser import SubgraphInfo, OperatorInfo
from ._operator_converter import OperatorConverter


class FXReconstructor:
    """
    Reconstructs TFLite execution graph as PyTorch FX graph.

    This class takes parsed TFLite graph information and converts it
    into a PyTorch FX graph that can be executed or exported.
    """

    def __init__(self, operator_converter: OperatorConverter | None = None):
        self.operator_converter = operator_converter or OperatorConverter()
        self.graph = torch.fx.Graph()
        self.tensor_map: dict[int, torch.fx.Node] = {}
        self.parameter_dict: dict[str, torch.Tensor] = {}
        self.node_counter = 0

    def reconstruct(
        self, subgraph: SubgraphInfo, weights: dict[int, torch.Tensor] | None = None
    ) -> torch.fx.GraphModule:
        """
        Reconstruct a TFLite subgraph as a PyTorch FX GraphModule.

        Args:
            subgraph: Parsed TFLite subgraph information
            weights: Optional dictionary mapping tensor indices to weight tensors

        Returns:
            PyTorch FX GraphModule representing the computation
        """
        weights = weights or {}
        self.graph = torch.fx.Graph()
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
        graph_module = torch.fx.GraphModule(root_module, self.graph)

        return graph_module

    def _process_operator(
        self,
        operator: OperatorInfo,
        subgraph: SubgraphInfo,
        weights: dict[int, torch.Tensor],
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
                param_name = self._sanitize_name(tensor_info.name or f"param_{input_idx}")
                self.parameter_dict[param_name] = weights[input_idx]
                param_node = self.graph.get_attr(param_name)
                self.tensor_map[input_idx] = param_node
                input_nodes.append(param_node)
            else:
                # Create a placeholder for missing weights/constants
                tensor_info = subgraph.tensors[input_idx]
                param_name = self._sanitize_name(tensor_info.name or f"const_{input_idx}")
                # Create dummy tensor based on shape
                shape = tensor_info.shape
                dummy_tensor = torch.zeros(shape)
                self.parameter_dict[param_name] = dummy_tensor
                param_node = self.graph.get_attr(param_name)
                self.tensor_map[input_idx] = param_node
                input_nodes.append(param_node)

        # Convert operator - now returns a callable that builds the FX graph
        try:
            graph_builder = self.operator_converter.convert(
                op_type, operator.inputs, operator.builtin_options
            )
        except NotImplementedError:
            # Create a placeholder for unsupported operators
            node_name = f"unsupported_{op_type.lower()}_{op_idx}"
            output_node = self.graph.call_function(
                lambda *args: args[0] if args else None,
                args=tuple(input_nodes) if input_nodes else (),
            )
            output_node.name = node_name
            for output_idx in operator.outputs:
                self.tensor_map[output_idx] = output_node
            return

        # Use the graph builder to create FX nodes
        # All conversion logic is now in the converter itself
        node_name = f"{op_type.lower()}_{op_idx}"
        node_counter_dict = {"count": self.node_counter}

        output_node = graph_builder(
            self.graph,
            input_nodes,
            weights,
            operator,
            subgraph,
            node_name,
            node_counter_dict,
            self.parameter_dict,
        )

        # Update node counter
        self.node_counter = node_counter_dict["count"]

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
                # Only register as parameter if it's a floating point or complex tensor
                # Integer tensors should be registered as buffers instead
                if value.dtype in (
                    torch.float32,
                    torch.float64,
                    torch.float16,
                    torch.complex64,
                    torch.complex128,
                ):
                    root.register_parameter(name, nn.Parameter(value))
                else:
                    root.register_buffer(name, value)
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
