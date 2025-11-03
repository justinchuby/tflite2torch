"""
TFLite to Torch operator conversion module.

This module provides mappings and conversions from TFLite operators
to their PyTorch equivalents. Each converter returns a function that
directly constructs FX graph nodes.
"""

from typing import Dict, List, Any, Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.fx import Graph, Node


class OperatorConverter:
    """
    Converts TFLite operators to PyTorch equivalents.

    This class maintains a registry of conversion functions that map
    TFLite operators to callables that directly construct FX graph nodes.
    All conversion logic is consolidated here - no more "custom" operators.
    """

    def __init__(self):
        self.converters: Dict[str, Callable] = {}
        self._register_converters()

    def _register_converters(self):
        """
        Register all operator converters.
        
        This registry includes all TFLite builtin operators from the official
        MLIR tfl_ops.td specification: https://www.tensorflow.org/mlir/tfl_ops
        """
        # Arithmetic & Math Operations
        self.converters["ABS"] = self._convert_abs
        self.converters["ADD"] = self._convert_add
        self.converters["ADD_N"] = self._convert_add_n
        self.converters["CEIL"] = self._convert_ceil
        self.converters["COS"] = self._convert_cos
        self.converters["DIV"] = self._convert_div
        self.converters["EXP"] = self._convert_exp
        self.converters["FLOOR"] = self._convert_floor
        self.converters["FLOOR_DIV"] = self._convert_floor_div
        self.converters["FLOOR_MOD"] = self._convert_floor_mod
        self.converters["LOG"] = self._convert_log
        self.converters["MAXIMUM"] = self._convert_maximum
        self.converters["MINIMUM"] = self._convert_minimum
        self.converters["MUL"] = self._convert_mul
        self.converters["NEG"] = self._convert_neg
        self.converters["POW"] = self._convert_pow
        self.converters["RSQRT"] = self._convert_rsqrt
        self.converters["SIN"] = self._convert_sin
        self.converters["SQRT"] = self._convert_sqrt
        self.converters["SQUARE"] = self._convert_square
        self.converters["SQUARED_DIFFERENCE"] = self._convert_squared_difference
        self.converters["SUB"] = self._convert_sub
        
        # Convolution & Pooling
        self.converters["AVERAGE_POOL_2D"] = self._convert_avg_pool2d
        self.converters["CONV_2D"] = self._convert_conv2d
        self.converters["CONV_3D"] = self._convert_conv3d
        self.converters["DEPTHWISE_CONV_2D"] = self._convert_depthwise_conv2d
        self.converters["MAX_POOL_2D"] = self._convert_max_pool2d
        self.converters["TRANSPOSE_CONV"] = self._convert_transpose_conv
        
        # Fully Connected
        self.converters["FULLY_CONNECTED"] = self._convert_fully_connected
        self.converters["BATCH_MATMUL"] = self._convert_batch_matmul
        
        # Activation Functions
        self.converters["ELU"] = self._convert_elu
        self.converters["GELU"] = self._convert_gelu
        self.converters["HARD_SWISH"] = self._convert_hard_swish
        self.converters["LEAKY_RELU"] = self._convert_leaky_relu
        self.converters["LOGISTIC"] = self._convert_sigmoid
        self.converters["LOG_SOFTMAX"] = self._convert_log_softmax
        self.converters["PRELU"] = self._convert_prelu
        self.converters["RELU"] = self._convert_relu
        self.converters["RELU6"] = self._convert_relu6
        self.converters["SOFTMAX"] = self._convert_softmax
        self.converters["TANH"] = self._convert_tanh
        
        # Normalization
        self.converters["L2_NORMALIZATION"] = self._convert_l2_normalization
        self.converters["LOCAL_RESPONSE_NORMALIZATION"] = self._convert_local_response_normalization
        
        # Reduction Operations
        self.converters["MEAN"] = self._convert_mean
        self.converters["REDUCE_MAX"] = self._convert_reduce_max
        self.converters["REDUCE_MIN"] = self._convert_reduce_min
        self.converters["REDUCE_PROD"] = self._convert_reduce_prod
        self.converters["REDUCE_ANY"] = self._convert_reduce_any
        self.converters["SUM"] = self._convert_sum
        
        # Shape & Tensor Manipulation
        self.converters["BATCH_TO_SPACE_ND"] = self._convert_batch_to_space
        self.converters["BROADCAST_ARGS"] = self._convert_broadcast_args
        self.converters["BROADCAST_TO"] = self._convert_broadcast_to
        self.converters["CONCATENATION"] = self._convert_concatenation
        self.converters["DEPTH_TO_SPACE"] = self._convert_depth_to_space
        self.converters["EXPAND_DIMS"] = self._convert_expand_dims
        self.converters["FILL"] = self._convert_fill
        self.converters["GATHER"] = self._convert_gather
        self.converters["GATHER_ND"] = self._convert_gather_nd
        self.converters["MIRROR_PAD"] = self._convert_mirror_pad
        self.converters["PACK"] = self._convert_pack
        self.converters["PAD"] = self._convert_pad
        self.converters["PADV2"] = self._convert_padv2
        self.converters["RANGE"] = self._convert_range
        self.converters["RESHAPE"] = self._convert_reshape
        self.converters["RESIZE_BILINEAR"] = self._convert_resize_bilinear
        self.converters["RESIZE_NEAREST_NEIGHBOR"] = self._convert_resize_nearest
        self.converters["REVERSE_V2"] = self._convert_reverse_v2
        self.converters["REVERSE_SEQUENCE"] = self._convert_reverse_sequence
        self.converters["SCATTER_ND"] = self._convert_scatter_nd
        self.converters["SHAPE"] = self._convert_shape
        self.converters["SLICE"] = self._convert_slice
        self.converters["SPACE_TO_BATCH_ND"] = self._convert_space_to_batch
        self.converters["SPACE_TO_DEPTH"] = self._convert_space_to_depth
        self.converters["SPARSE_TO_DENSE"] = self._convert_sparse_to_dense
        self.converters["SPLIT"] = self._convert_split
        self.converters["SPLIT_V"] = self._convert_split_v
        self.converters["SQUEEZE"] = self._convert_squeeze
        self.converters["STRIDED_SLICE"] = self._convert_strided_slice
        self.converters["TILE"] = self._convert_tile
        self.converters["TOPK_V2"] = self._convert_topk_v2
        self.converters["TRANSPOSE"] = self._convert_transpose
        self.converters["UNPACK"] = self._convert_unpack
        self.converters["UNIQUE"] = self._convert_unique
        self.converters["WHERE"] = self._convert_where
        self.converters["ZEROS_LIKE"] = self._convert_zeros_like
        
        # Comparison Operations
        self.converters["EQUAL"] = self._convert_equal
        self.converters["GREATER"] = self._convert_greater
        self.converters["GREATER_EQUAL"] = self._convert_greater_equal
        self.converters["LESS"] = self._convert_less
        self.converters["LESS_EQUAL"] = self._convert_less_equal
        self.converters["NOT_EQUAL"] = self._convert_not_equal
        
        # Logical Operations
        self.converters["LOGICAL_AND"] = self._convert_logical_and
        self.converters["LOGICAL_NOT"] = self._convert_logical_not
        self.converters["LOGICAL_OR"] = self._convert_logical_or
        
        # Selection Operations
        self.converters["ARG_MAX"] = self._convert_arg_max
        self.converters["ARG_MIN"] = self._convert_arg_min
        self.converters["ONE_HOT"] = self._convert_one_hot
        self.converters["SELECT"] = self._convert_select
        self.converters["SELECT_V2"] = self._convert_select_v2
        
        # Recurrent Neural Network Operations
        self.converters["LSTM"] = self._convert_lstm
        self.converters["BIDIRECTIONAL_SEQUENCE_LSTM"] = self._convert_bidirectional_sequence_lstm
        self.converters["UNIDIRECTIONAL_SEQUENCE_LSTM"] = self._convert_unidirectional_sequence_lstm
        self.converters["RNN"] = self._convert_rnn
        self.converters["BIDIRECTIONAL_SEQUENCE_RNN"] = self._convert_bidirectional_sequence_rnn
        self.converters["UNIDIRECTIONAL_SEQUENCE_RNN"] = self._convert_unidirectional_sequence_rnn
        
        # Quantization Operations
        self.converters["QUANTIZE"] = self._convert_quantize
        self.converters["DEQUANTIZE"] = self._convert_dequantize
        self.converters["FAKE_QUANT"] = self._convert_fake_quant
        
        # Type Conversion
        self.converters["CAST"] = self._convert_cast
        
        # Embedding & Lookup
        self.converters["EMBEDDING_LOOKUP"] = self._convert_embedding_lookup
        self.converters["HASHTABLE_LOOKUP"] = self._convert_hashtable_lookup
        
        # Custom & Advanced Operations
        self.converters["CUSTOM"] = self._convert_custom
        self.converters["CUMSUM"] = self._convert_cumsum
        self.converters["MATRIX_DIAG"] = self._convert_matrix_diag
        self.converters["MATRIX_SET_DIAG"] = self._convert_matrix_set_diag
        self.converters["SEGMENT_SUM"] = self._convert_segment_sum
        
        # Signal Processing Operations
        self.converters["RFFT2D"] = self._convert_rfft2d

    def convert(
        self, op_type: str, inputs: List[Any], options: Dict[str, Any]
    ) -> Callable:
        """
        Convert a TFLite operator to a graph construction function.

        Args:
            op_type: TFLite operator type
            inputs: List of input specifications
            options: Operator-specific options

        Returns:
            A callable that takes (graph: Graph, input_nodes: List[Node], 
            weights: Dict, operator, subgraph, node_name: str, node_counter: Dict,
            parameter_dict: Dict) and returns a Node or tuple of Nodes.
        """
        if op_type not in self.converters:
            raise NotImplementedError(f"Operator {op_type} is not supported yet")

        converter_result = self.converters[op_type](inputs, options)
        
        # If it's already a callable (new format), return it directly
        if callable(converter_result):
            return converter_result
        
        # Otherwise, it's the old dict format - wrap it in a callable
        return self._wrap_legacy_converter(converter_result, op_type, inputs, options)
    
    def _wrap_legacy_converter(self, conv_info: Dict[str, Any], op_type: str, 
                               inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """
        Wrap legacy dict-based converter results in a callable that builds FX graph nodes.
        This allows gradual migration to the new format.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph from legacy converter info."""
            from ._fx_reconstructor import FXReconstructor
            
            # Delegate to _fx_reconstructor's logic for now
            # This is a temporary bridge during migration
            reconstructor = FXReconstructor()
            reconstructor.graph = graph
            reconstructor.node_counter = node_counter['count']
            reconstructor.parameter_dict = parameter_dict
            
            # Use the existing logic from FXReconstructor._process_operator
            # but adapted for this context
            if conv_info.get("custom", False):
                # Use the custom operator handling from FXReconstructor
                module_name = conv_info.get("module", "unknown")
                output_node = reconstructor._handle_custom_operator(
                    module_name, input_nodes, operator, subgraph, weights, node_name
                )
            elif isinstance(conv_info.get("module"), type) and issubclass(conv_info["module"], nn.Module):
                # Module class - create instance and add to graph
                params = conv_info.get("params", {})
                # Infer parameters from operator if needed
                # (This logic should be in the converter itself, but for legacy support...)
                if conv_info["module"] == nn.Conv2d and len(operator.inputs) >= 2:
                    weight_idx = operator.inputs[1]
                    if weight_idx < len(subgraph.tensors):
                        weight_info = subgraph.tensors[weight_idx]
                        if len(weight_info.shape) == 4:
                            params.setdefault("out_channels", weight_info.shape[0])
                            params.setdefault("kernel_size", (weight_info.shape[1], weight_info.shape[2]))
                            params.setdefault("in_channels", weight_info.shape[3])
                            if len(operator.inputs) >= 3 and operator.inputs[2] >= 0:
                                params.setdefault("bias", True)
                            else:
                                params.setdefault("bias", False)
                
                module = conv_info["module"](**params)
                
                # Load weights
                if len(operator.inputs) >= 2:
                    weight_idx = operator.inputs[1]
                    if weight_idx in weights:
                        weight_tensor = weights[weight_idx]
                        if conv_info["module"] == nn.Conv2d:
                            weight_tensor = weight_tensor.permute(0, 3, 1, 2)
                        module.weight.data = weight_tensor
                    
                    if len(operator.inputs) >= 3:
                        bias_idx = operator.inputs[2]
                        if bias_idx >= 0 and bias_idx in weights and hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data = weights[bias_idx]
                
                module_name_str = f"module_{node_counter['count']}"
                node_counter['count'] += 1
                parameter_dict[module_name_str] = module
                output_node = graph.call_module(module_name_str, args=(input_nodes[0],) if input_nodes else ())
                output_node.name = node_name
                
                # Handle activation
                if "activation" in conv_info and conv_info["activation"] != "NONE":
                    activation_module = self.get_activation_module(conv_info["activation"])
                    if activation_module:
                        act_name = f"activation_{node_counter['count']}"
                        node_counter['count'] += 1
                        parameter_dict[act_name] = activation_module
                        output_node = graph.call_module(act_name, args=(output_node,))
                        output_node.name = f"{node_name}_activation"
            else:
                # Function - create call_function node
                target = conv_info.get("module")
                params = conv_info.get("params", {})
                output_node = graph.call_function(target, args=tuple(input_nodes), kwargs=params)
                output_node.name = node_name
                
                # Handle activation
                if "activation" in conv_info and conv_info["activation"] != "NONE":
                    activation_module = self.get_activation_module(conv_info["activation"])
                    if activation_module:
                        act_name = f"activation_{node_counter['count']}"
                        node_counter['count'] += 1
                        parameter_dict[act_name] = activation_module
                        output_node = graph.call_module(act_name, args=(output_node,))
                        output_node.name = f"{node_name}_activation"
            
            return output_node
        
        return build_graph

    def _simple_call_function(self, target: Callable, with_activation: bool = False) -> Callable:
        """Helper to create a simple call_function converter."""
        def converter(inputs: List[Any], options: Dict[str, Any]) -> Callable:
            activation = options.get("fused_activation_function", "NONE") if with_activation else "NONE"
            
            def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                           operator, subgraph, node_name: str, node_counter: Dict,
                           parameter_dict: Dict) -> Node:
                output_node = graph.call_function(target, args=tuple(input_nodes))
                output_node.name = node_name
                
                if activation != "NONE":
                    activation_module = self.get_activation_module(activation)
                    if activation_module is not None:
                        act_name = f"activation_{node_counter['count']}"
                        node_counter['count'] += 1
                        parameter_dict[act_name] = activation_module
                        output_node = graph.call_module(act_name, args=(output_node,))
                        output_node.name = f"{node_name}_activation"
                
                return output_node
            return build_graph
        return converter

    def _simple_call_module(self, module_class: type, **default_params) -> Callable:
        """Helper to create a simple call_module converter."""
        def converter(inputs: List[Any], options: Dict[str, Any]) -> Callable:
            params = {**default_params, **{k: v for k, v in options.items() if k in default_params}}
            
            def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                           operator, subgraph, node_name: str, node_counter: Dict,
                           parameter_dict: Dict) -> Node:
                module = module_class(**params)
                module_name = f"module_{node_counter['count']}"
                node_counter['count'] += 1
                parameter_dict[module_name] = module
                output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
                output_node.name = node_name
                return output_node
            return build_graph
        return converter

    def _convert_conv2d(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite CONV_2D to PyTorch Conv2d."""
        # Extract parameters from options
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        padding = options.get("padding", "SAME")
        dilation_h = options.get("dilation_h_factor", 1)
        dilation_w = options.get("dilation_w_factor", 1)
        activation = options.get("fused_activation_function", "NONE")

        # Convert padding from TFLite to PyTorch format
        if padding == "SAME":
            padding_mode = "same"
        elif padding == "VALID":
            padding_mode = 0
        else:
            padding_mode = 0

        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph nodes for Conv2d."""
            # Extract weight tensor shape to determine conv parameters
            if len(operator.inputs) >= 2:
                weight_idx = operator.inputs[1]
                weight_tensor_info = subgraph.tensors[weight_idx]
                # TFLite weight format: [out_channels, kernel_h, kernel_w, in_channels]
                if len(weight_tensor_info.shape) == 4:
                    out_channels = weight_tensor_info.shape[0]
                    kernel_size = (weight_tensor_info.shape[1], weight_tensor_info.shape[2])
                    in_channels = weight_tensor_info.shape[3]
                    
                    # Check if bias exists
                    has_bias = len(operator.inputs) >= 3 and operator.inputs[2] >= 0
                    
                    # Create Conv2d module
                    params = {
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "kernel_size": kernel_size,
                        "stride": (stride_h, stride_w),
                        "padding": padding_mode,
                        "dilation": (dilation_h, dilation_w),
                        "bias": has_bias
                    }
                    module = nn.Conv2d(**params)
                    
                    # Load weights
                    if weight_idx in weights:
                        weight_tensor = weights[weight_idx]
                        # Convert from TFLite format to PyTorch format
                        # TFLite: [out_channels, kernel_h, kernel_w, in_channels]
                        # PyTorch: [out_channels, in_channels, kernel_h, kernel_w]
                        weight_tensor = weight_tensor.permute(0, 3, 1, 2)
                        module.weight.data = weight_tensor
                    
                    # Load bias if exists
                    if has_bias:
                        bias_idx = operator.inputs[2]
                        if bias_idx in weights and module.bias is not None:
                            module.bias.data = weights[bias_idx]
                    
                    # Add module to parameter dict
                    module_name = f"module_{node_counter['count']}"
                    node_counter['count'] += 1
                    parameter_dict[module_name] = module
                    
                    # Create call_module node
                    output_node = graph.call_module(module_name, args=(input_nodes[0],))
                    output_node.name = node_name
                    
                    # Handle fused activation
                    if activation != "NONE":
                        activation_module = self.get_activation_module(activation)
                        if activation_module is not None:
                            act_name = f"activation_{node_counter['count']}"
                            node_counter['count'] += 1
                            parameter_dict[act_name] = activation_module
                            output_node = graph.call_module(act_name, args=(output_node,))
                            output_node.name = f"{node_name}_activation"
                    
                    return output_node
            
            # Fallback: create a placeholder node
            return graph.call_function(lambda x: x, args=(input_nodes[0],) if input_nodes else ())
        
        return build_graph

    def _convert_depthwise_conv2d(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite DEPTHWISE_CONV_2D to PyTorch depthwise Conv2d."""
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        padding = options.get("padding", "SAME")
        depth_multiplier = options.get("depth_multiplier", 1)

        if padding == "SAME":
            padding_mode = "same"
        else:
            padding_mode = 0

        return {
            "module": nn.Conv2d,
            "params": {
                "stride": (stride_h, stride_w),
                "padding": padding_mode,
                "groups": -1,  # Will be set to in_channels
            },
            "activation": options.get("fused_activation_function", "NONE"),
            "depthwise": True,
        }

    def _convert_fully_connected(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite FULLY_CONNECTED to PyTorch Linear."""
        return {
            "module": nn.Linear,
            "params": {},
            "activation": options.get("fused_activation_function", "NONE"),
        }

    def _convert_add(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite ADD to PyTorch addition."""
        activation = options.get("fused_activation_function", "NONE")
        
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph nodes for ADD."""
            output_node = graph.call_function(torch.add, args=tuple(input_nodes))
            output_node.name = node_name
            
            # Handle fused activation
            if activation != "NONE":
                activation_module = self.get_activation_module(activation)
                if activation_module is not None:
                    act_name = f"activation_{node_counter['count']}"
                    node_counter['count'] += 1
                    parameter_dict[act_name] = activation_module
                    output_node = graph.call_module(act_name, args=(output_node,))
                    output_node.name = f"{node_name}_activation"
            
            return output_node
        
        return build_graph

    def _convert_mul(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite MUL to PyTorch multiplication."""
        activation = options.get("fused_activation_function", "NONE")
        
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            output_node = graph.call_function(torch.mul, args=tuple(input_nodes))
            output_node.name = node_name
            
            if activation != "NONE":
                activation_module = self.get_activation_module(activation)
                if activation_module is not None:
                    act_name = f"activation_{node_counter['count']}"
                    node_counter['count'] += 1
                    parameter_dict[act_name] = activation_module
                    output_node = graph.call_module(act_name, args=(output_node,))
                    output_node.name = f"{node_name}_activation"
            
            return output_node
        
        return build_graph

    def _convert_sub(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SUB to PyTorch subtraction."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            output_node = graph.call_function(torch.sub, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_div(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite DIV to PyTorch division."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            output_node = graph.call_function(torch.div, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_relu(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite RELU to PyTorch ReLU."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            module = nn.ReLU()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],))
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_relu6(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite RELU6 to PyTorch ReLU6."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            module = nn.ReLU6()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],))
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_tanh(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite TANH to PyTorch Tanh."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            module = nn.Tanh()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],))
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_sigmoid(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LOGISTIC to PyTorch Sigmoid."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            module = nn.Sigmoid()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],))
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_softmax(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SOFTMAX to PyTorch Softmax."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            module = nn.Softmax(dim=-1)
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],))
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_max_pool2d(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite MAX_POOL_2D to PyTorch MaxPool2d."""
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        filter_height = options.get("filter_height", 2)
        filter_width = options.get("filter_width", 2)
        padding = options.get("padding", "VALID")

        if padding == "SAME":
            padding_mode = "same"
        else:
            padding_mode = 0

        return {
            "module": nn.MaxPool2d,
            "params": {
                "kernel_size": (filter_height, filter_width),
                "stride": (stride_h, stride_w),
                "padding": padding_mode,
            },
        }

    def _convert_avg_pool2d(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite AVERAGE_POOL_2D to PyTorch AvgPool2d."""
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        filter_height = options.get("filter_height", 2)
        filter_width = options.get("filter_width", 2)
        padding = options.get("padding", "VALID")

        if padding == "SAME":
            padding_mode = "same"
        else:
            padding_mode = 0

        return {
            "module": nn.AvgPool2d,
            "params": {
                "kernel_size": (filter_height, filter_width),
                "stride": (stride_h, stride_w),
                "padding": padding_mode,
            },
        }

    def _convert_reshape(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite RESHAPE to PyTorch reshape.
        
        RESHAPE takes 2 inputs: input tensor and shape tensor.
        The shape tensor needs to be converted to a tuple.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for RESHAPE."""
            if len(input_nodes) >= 2:
                input_node = input_nodes[0]
                shape_idx = operator.inputs[1]
                if shape_idx in weights:
                    shape_tensor = weights[shape_idx]
                    shape_tuple = tuple(shape_tensor.tolist())
                    output_node = graph.call_function(
                        torch.reshape,
                        args=(input_node, shape_tuple)
                    )
                else:
                    # If shape is not available as constant, use -1 for unknown dimension
                    output_node = graph.call_function(
                        torch.reshape,
                        args=(input_node, (-1,))
                    )
            else:
                output_node = graph.call_function(
                    lambda x: x,
                    args=(input_nodes[0],) if input_nodes else ()
                )
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_concatenation(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Callable:
        """Convert TFLite CONCATENATION to PyTorch cat.
        
        CONCATENATION takes multiple input tensors and concatenates them.
        The axis is specified in options.
        """
        axis = options.get("axis", 0)
        
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for CONCATENATION."""
            if len(input_nodes) > 0:
                output_node = graph.call_function(
                    torch.cat,
                    args=(tuple(input_nodes),),
                    kwargs={"dim": axis}
                )
            else:
                output_node = graph.call_function(
                    lambda *args: args[0] if args else None,
                    args=()
                )
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_transpose(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite TRANSPOSE to PyTorch permute.
        
        TRANSPOSE takes 2 inputs: input tensor and perm tensor.
        The perm tensor needs to be converted to a tuple.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for TRANSPOSE."""
            if len(input_nodes) >= 2:
                input_node = input_nodes[0]
                perm_idx = operator.inputs[1]
                if perm_idx in weights:
                    perm_tensor = weights[perm_idx]
                    perm_tuple = tuple(perm_tensor.tolist())
                    output_node = graph.call_function(
                        torch.permute,
                        args=(input_node, perm_tuple)
                    )
                else:
                    # If perm is not available, pass through
                    output_node = graph.call_function(
                        lambda x: x,
                        args=(input_node,)
                    )
            else:
                output_node = graph.call_function(
                    lambda x: x,
                    args=(input_nodes[0],) if input_nodes else ()
                )
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_mean(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite MEAN to PyTorch mean.
        
        MEAN takes 2 inputs: input tensor and reduction_indices tensor.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict, 
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for MEAN."""
            if len(input_nodes) >= 2:
                input_node = input_nodes[0]
                axis_idx = operator.inputs[1]
                keep_dims = operator.builtin_options.get("keep_dims", False)
                
                # Infer keep_dims from output shape if not in options
                if not keep_dims and len(operator.outputs) > 0:
                    input_idx = operator.inputs[0]
                    output_idx = operator.outputs[0]
                    if input_idx < len(subgraph.tensors) and output_idx < len(subgraph.tensors):
                        input_shape = subgraph.tensors[input_idx].shape
                        output_shape = subgraph.tensors[output_idx].shape
                        if len(input_shape) == len(output_shape):
                            keep_dims = True
                
                if axis_idx in weights:
                    axis_tensor = weights[axis_idx]
                    axis_list = axis_tensor.tolist()
                    if isinstance(axis_list, list):
                        axis = tuple(axis_list) if len(axis_list) > 1 else axis_list[0]
                    else:
                        axis = axis_list
                    
                    output_node = graph.call_function(
                        torch.mean,
                        args=(input_node,),
                        kwargs={"dim": axis, "keepdim": keep_dims}
                    )
                else:
                    # If axis is not available, reduce all dimensions
                    output_node = graph.call_function(torch.mean, args=(input_node,))
            else:
                output_node = graph.call_function(lambda x: x, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_pad(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite PAD to PyTorch pad.
        
        PAD takes 2 inputs: input tensor and paddings tensor.
        The paddings tensor needs to be converted to a tuple.
        """
        return {
            "module": "pad",
            "params": {},
            "custom": True,
        }

    def _convert_squeeze(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SQUEEZE to PyTorch squeeze."""
        squeeze_dims = options.get("squeeze_dims", None)
        return {
            "module": torch.squeeze,
            "params": {"dim": squeeze_dims},
        }

    def _convert_expand_dims(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite EXPAND_DIMS to PyTorch unsqueeze."""
        return {
            "module": torch.unsqueeze,
            "params": {},
        }

    def _convert_slice(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SLICE to PyTorch slice indexing."""
        return {
            "module": torch.slice,
            "params": {},
        }

    def _convert_gather(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite GATHER to PyTorch gather."""
        axis = options.get("axis", 0)
        return {
            "module": torch.gather,
            "params": {"dim": axis},
        }

    def _convert_split(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SPLIT to PyTorch split."""
        num_splits = options.get("num_splits", 1)
        return {
            "module": torch.split,
            "params": {"split_size_or_sections": num_splits},
        }

    def _convert_batch_to_space(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite BATCH_TO_SPACE_ND to PyTorch operations."""
        return {
            "module": "batch_to_space",
            "params": {},
            "custom": True,
        }

    def _convert_space_to_batch(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite SPACE_TO_BATCH_ND to PyTorch operations."""
        return {
            "module": "space_to_batch",
            "params": {},
            "custom": True,
        }

    def _convert_resize_bilinear(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite RESIZE_BILINEAR to PyTorch interpolate."""
        align_corners = options.get("align_corners", False)
        return {
            "module": nn.functional.interpolate,
            "params": {
                "mode": "bilinear",
                "align_corners": align_corners,
            },
        }

    def _convert_resize_nearest(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite RESIZE_NEAREST_NEIGHBOR to PyTorch interpolate."""
        return {
            "module": nn.functional.interpolate,
            "params": {
                "mode": "nearest",
            },
        }

    # Additional Arithmetic & Math Operations
    def _convert_abs(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite ABS to PyTorch abs."""
        return {"module": torch.abs, "params": {}}
    
    def _convert_add_n(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite ADD_N to PyTorch sum of tensors."""
        return {"module": torch.sum, "params": {"dim": 0}}
    
    def _convert_ceil(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite CEIL to PyTorch ceil."""
        return {"module": torch.ceil, "params": {}}
    
    def _convert_cos(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite COS to PyTorch cos."""
        return {"module": torch.cos, "params": {}}
    
    def _convert_exp(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite EXP to PyTorch exp."""
        return {"module": torch.exp, "params": {}}
    
    def _convert_floor(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite FLOOR to PyTorch floor."""
        return {"module": torch.floor, "params": {}}
    
    def _convert_floor_div(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite FLOOR_DIV to PyTorch floor_divide."""
        return {"module": torch.floor_divide, "params": {}}
    
    def _convert_floor_mod(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite FLOOR_MOD to PyTorch fmod."""
        return {"module": torch.fmod, "params": {}}
    
    def _convert_log(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LOG to PyTorch log."""
        return {"module": torch.log, "params": {}}
    
    def _convert_maximum(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite MAXIMUM to PyTorch maximum."""
        return {"module": torch.maximum, "params": {}}
    
    def _convert_minimum(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite MINIMUM to PyTorch minimum."""
        return {"module": torch.minimum, "params": {}}
    
    def _convert_neg(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite NEG to PyTorch neg."""
        return {"module": torch.neg, "params": {}}
    
    def _convert_pow(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite POW to PyTorch pow."""
        return {"module": torch.pow, "params": {}}
    
    def _convert_rsqrt(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite RSQRT to PyTorch rsqrt."""
        return {"module": torch.rsqrt, "params": {}}
    
    def _convert_sin(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SIN to PyTorch sin."""
        return {"module": torch.sin, "params": {}}
    
    def _convert_square(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SQUARE to PyTorch square."""
        return {"module": torch.square, "params": {}}
    
    def _convert_squared_difference(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SQUARED_DIFFERENCE to PyTorch operations."""
        return {"module": "squared_difference", "params": {}, "custom": True}
    
    def _convert_sqrt(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SQRT to PyTorch sqrt."""
        return {"module": torch.sqrt, "params": {}}
    
    # Additional Convolution Operations
    def _convert_conv3d(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite CONV_3D to PyTorch Conv3d."""
        stride = options.get("stride", [1, 1, 1])
        padding = options.get("padding", "VALID")
        return {
            "module": nn.Conv3d,
            "params": {
                "stride": stride,
                "padding": 0 if padding == "VALID" else "same",
            },
            "activation": options.get("fused_activation_function", "NONE"),
        }
    
    def _convert_transpose_conv(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite TRANSPOSE_CONV to PyTorch ConvTranspose2d."""
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        padding = options.get("padding", "SAME")
        return {
            "module": nn.ConvTranspose2d,
            "params": {
                "stride": (stride_h, stride_w),
                "padding": 0 if padding == "VALID" else "same",
            },
            "activation": options.get("fused_activation_function", "NONE"),
        }
    
    def _convert_batch_matmul(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite BATCH_MATMUL to PyTorch bmm."""
        return {"module": torch.bmm, "params": {}}
    
    # Additional Activation Functions
    def _convert_elu(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite ELU to PyTorch ELU."""
        return {"module": nn.ELU, "params": {}}
    
    def _convert_gelu(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite GELU to PyTorch GELU."""
        return {"module": nn.GELU, "params": {}}
    
    def _convert_hard_swish(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite HARD_SWISH to PyTorch Hardswish."""
        return {"module": nn.Hardswish, "params": {}}
    
    def _convert_leaky_relu(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LEAKY_RELU to PyTorch LeakyReLU."""
        alpha = options.get("alpha", 0.01)
        return {"module": nn.LeakyReLU, "params": {"negative_slope": alpha}}
    
    def _convert_log_softmax(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LOG_SOFTMAX to PyTorch LogSoftmax."""
        return {"module": nn.LogSoftmax, "params": {"dim": -1}}
    
    def _convert_prelu(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite PRELU to PyTorch PReLU."""
        return {"module": nn.PReLU, "params": {}}
    
    # Normalization Operations
    def _convert_l2_normalization(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite L2_NORMALIZATION to PyTorch normalize."""
        return {"module": torch.nn.functional.normalize, "params": {"p": 2, "dim": -1}}
    
    def _convert_local_response_normalization(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LOCAL_RESPONSE_NORMALIZATION to PyTorch LocalResponseNorm."""
        size = options.get("radius", 5) * 2 + 1
        return {"module": nn.LocalResponseNorm, "params": {"size": size}}
    
    # Additional Reduction Operations
    def _convert_reduce_max(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite REDUCE_MAX to PyTorch max.
        
        REDUCE_MAX takes 2 inputs: input tensor and reduction_indices tensor.
        """
        keep_dims = options.get("keep_dims", False)
        return {"module": "reduce_max", "params": {"keep_dims": keep_dims}, "custom": True}
    
    def _convert_reduce_min(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite REDUCE_MIN to PyTorch min.
        
        REDUCE_MIN takes 2 inputs: input tensor and reduction_indices tensor.
        """
        keep_dims = options.get("keep_dims", False)
        return {"module": "reduce_min", "params": {"keep_dims": keep_dims}, "custom": True}
    
    def _convert_reduce_prod(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite REDUCE_PROD to PyTorch prod."""
        keep_dims = options.get("keep_dims", False)
        return {"module": torch.prod, "params": {"keepdim": keep_dims}}
    
    def _convert_reduce_any(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite REDUCE_ANY to PyTorch any."""
        keep_dims = options.get("keep_dims", False)
        return {"module": torch.any, "params": {"keepdim": keep_dims}}
    
    def _convert_sum(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SUM to PyTorch sum.
        
        SUM takes 2 inputs: input tensor and reduction_indices tensor.
        """
        keep_dims = options.get("keep_dims", False)
        return {"module": "sum", "params": {"keep_dims": keep_dims}, "custom": True}
    
    # Additional Shape & Tensor Manipulation
    def _convert_broadcast_args(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite BROADCAST_ARGS to PyTorch broadcast_shapes."""
        return {"module": torch.broadcast_shapes, "params": {}}
    
    def _convert_broadcast_to(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite BROADCAST_TO to PyTorch broadcast_to."""
        return {"module": torch.broadcast_to, "params": {}}
    
    def _convert_depth_to_space(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite DEPTH_TO_SPACE to PyTorch pixel_shuffle."""
        block_size = options.get("block_size", 2)
        return {"module": nn.PixelShuffle, "params": {"upscale_factor": block_size}}
    
    def _convert_fill(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite FILL to PyTorch full."""
        return {"module": torch.full, "params": {}}
    
    def _convert_gather_nd(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite GATHER_ND to PyTorch operations."""
        return {"module": "gather_nd", "params": {}, "custom": True}
    
    def _convert_mirror_pad(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite MIRROR_PAD to PyTorch pad with reflect mode."""
        mode = options.get("mode", "REFLECT")
        return {"module": torch.nn.functional.pad, "params": {"mode": "reflect"}}
    
    def _convert_pack(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite PACK to PyTorch stack.
        
        PACK takes multiple input tensors and stacks them along a new dimension.
        PyTorch stack expects tensors as a sequence (tuple/list).
        """
        axis = options.get("axis", 0)
        return {"module": "pack", "params": {"axis": axis}, "custom": True}
    
    def _convert_padv2(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite PADV2 to PyTorch pad."""
        return {"module": torch.nn.functional.pad, "params": {}}
    
    def _convert_range(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite RANGE to PyTorch arange."""
        return {"module": torch.arange, "params": {}}
    
    def _convert_reverse_v2(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite REVERSE_V2 to PyTorch flip."""
        return {"module": torch.flip, "params": {}}
    
    def _convert_reverse_sequence(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite REVERSE_SEQUENCE to PyTorch operations."""
        return {"module": "reverse_sequence", "params": {}, "custom": True}
    
    def _convert_scatter_nd(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SCATTER_ND to PyTorch scatter."""
        return {"module": torch.scatter, "params": {}}
    
    def _convert_shape(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SHAPE to PyTorch shape property."""
        return {"module": "shape", "params": {}, "custom": True}
    
    def _convert_space_to_depth(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SPACE_TO_DEPTH to PyTorch pixel_unshuffle."""
        block_size = options.get("block_size", 2)
        return {"module": nn.PixelUnshuffle, "params": {"downscale_factor": block_size}}
    
    def _convert_sparse_to_dense(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SPARSE_TO_DENSE to PyTorch operations."""
        return {"module": "sparse_to_dense", "params": {}, "custom": True}
    
    def _convert_split_v(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SPLIT_V to PyTorch split."""
        return {"module": torch.split, "params": {}}
    
    def _convert_strided_slice(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite STRIDED_SLICE to PyTorch slice."""
        return {"module": "strided_slice", "params": {}, "custom": True}
    
    def _convert_tile(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite TILE to PyTorch tile."""
        return {"module": torch.tile, "params": {}}
    
    def _convert_topk_v2(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite TOPK_V2 to PyTorch topk."""
        k = options.get("k", 1)
        return {"module": torch.topk, "params": {"k": k}}
    
    def _convert_unpack(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite UNPACK to PyTorch unbind."""
        axis = options.get("axis", 0)
        return {"module": torch.unbind, "params": {"dim": axis}}
    
    def _convert_unique(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite UNIQUE to PyTorch unique."""
        return {"module": torch.unique, "params": {}}
    
    def _convert_where(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite WHERE to PyTorch where."""
        return {"module": torch.where, "params": {}}
    
    def _convert_zeros_like(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite ZEROS_LIKE to PyTorch zeros_like."""
        return {"module": torch.zeros_like, "params": {}}
    
    # Comparison Operations
    def _convert_equal(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite EQUAL to PyTorch eq."""
        return {"module": torch.eq, "params": {}}
    
    def _convert_greater(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite GREATER to PyTorch gt."""
        return {"module": torch.gt, "params": {}}
    
    def _convert_greater_equal(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite GREATER_EQUAL to PyTorch ge."""
        return {"module": torch.ge, "params": {}}
    
    def _convert_less(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LESS to PyTorch lt."""
        return {"module": torch.lt, "params": {}}
    
    def _convert_less_equal(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LESS_EQUAL to PyTorch le."""
        return {"module": torch.le, "params": {}}
    
    def _convert_not_equal(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite NOT_EQUAL to PyTorch ne."""
        return {"module": torch.ne, "params": {}}
    
    # Logical Operations
    def _convert_logical_and(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LOGICAL_AND to PyTorch logical_and."""
        return {"module": torch.logical_and, "params": {}}
    
    def _convert_logical_not(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LOGICAL_NOT to PyTorch logical_not."""
        return {"module": torch.logical_not, "params": {}}
    
    def _convert_logical_or(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LOGICAL_OR to PyTorch logical_or."""
        return {"module": torch.logical_or, "params": {}}
    
    # Selection Operations
    def _convert_arg_max(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite ARG_MAX to PyTorch argmax."""
        return {"module": torch.argmax, "params": {}}
    
    def _convert_arg_min(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite ARG_MIN to PyTorch argmin."""
        return {"module": torch.argmin, "params": {}}
    
    def _convert_one_hot(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite ONE_HOT to PyTorch one_hot."""
        return {"module": torch.nn.functional.one_hot, "params": {}}
    
    def _convert_select(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SELECT to PyTorch where."""
        return {"module": torch.where, "params": {}}
    
    def _convert_select_v2(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SELECT_V2 to PyTorch where."""
        return {"module": torch.where, "params": {}}
    
    # Recurrent Neural Network Operations
    def _convert_lstm(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LSTM to PyTorch LSTM."""
        return {"module": nn.LSTM, "params": {}}
    
    def _convert_bidirectional_sequence_lstm(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite BIDIRECTIONAL_SEQUENCE_LSTM to PyTorch LSTM."""
        return {"module": nn.LSTM, "params": {"bidirectional": True}}
    
    def _convert_unidirectional_sequence_lstm(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite UNIDIRECTIONAL_SEQUENCE_LSTM to PyTorch LSTM."""
        return {"module": nn.LSTM, "params": {}}
    
    def _convert_rnn(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite RNN to PyTorch RNN."""
        return {"module": nn.RNN, "params": {}}
    
    def _convert_bidirectional_sequence_rnn(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite BIDIRECTIONAL_SEQUENCE_RNN to PyTorch RNN."""
        return {"module": nn.RNN, "params": {"bidirectional": True}}
    
    def _convert_unidirectional_sequence_rnn(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite UNIDIRECTIONAL_SEQUENCE_RNN to PyTorch RNN."""
        return {"module": nn.RNN, "params": {}}
    
    # Quantization Operations
    def _convert_quantize(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite QUANTIZE to PyTorch quantize_per_tensor."""
        return {"module": torch.quantize_per_tensor, "params": {}}
    
    def _convert_dequantize(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite DEQUANTIZE to PyTorch dequantize."""
        return {"module": "dequantize", "params": {}, "custom": True}
    
    def _convert_fake_quant(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite FAKE_QUANT to PyTorch fake_quantize_per_tensor_affine."""
        return {"module": torch.fake_quantize_per_tensor_affine, "params": {}}
    
    # Type Conversion
    def _convert_cast(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite CAST to PyTorch to/type conversion."""
        return {"module": "cast", "params": {}, "custom": True}
    
    # Embedding & Lookup
    def _convert_embedding_lookup(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite EMBEDDING_LOOKUP to PyTorch Embedding."""
        return {"module": nn.Embedding, "params": {}}
    
    def _convert_hashtable_lookup(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite HASHTABLE_LOOKUP to custom implementation."""
        return {"module": "hashtable_lookup", "params": {}, "custom": True}
    
    # Custom & Advanced Operations
    def _convert_custom(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite CUSTOM operation."""
        return {"module": "custom", "params": {}, "custom": True}
    
    def _convert_cumsum(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite CUMSUM to PyTorch cumsum."""
        return {"module": torch.cumsum, "params": {}}
    
    def _convert_matrix_diag(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite MATRIX_DIAG to PyTorch diag."""
        return {"module": torch.diag, "params": {}}
    
    def _convert_matrix_set_diag(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite MATRIX_SET_DIAG to custom implementation."""
        return {"module": "matrix_set_diag", "params": {}, "custom": True}
    
    def _convert_segment_sum(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SEGMENT_SUM to PyTorch segment operations."""
        return {"module": "segment_sum", "params": {}, "custom": True}
    
    # Signal Processing Operations
    def _convert_rfft2d(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert TFLite RFFT2D to PyTorch fft.rfft2.
        
        RFFT2D takes 2 inputs:
        - input tensor (signal)
        - fft_length tensor (shape [2])
        
        The fft_length needs to be converted to a tuple for PyTorch.
        """
        return {"module": "rfft2d", "params": {}, "custom": True}

    def get_activation_module(self, activation: str) -> Optional[nn.Module]:
        """
        Get PyTorch activation module for a TFLite fused activation.

        Args:
            activation: TFLite activation function name

        Returns:
            PyTorch activation module or None
        """
        activation_map = {
            "NONE": None,
            "RELU": nn.ReLU(),
            "RELU6": nn.ReLU6(),
            "TANH": nn.Tanh(),
            "SIGN_BIT": None,
        }
        return activation_map.get(activation, None)
