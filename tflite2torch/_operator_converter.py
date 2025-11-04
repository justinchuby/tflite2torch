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
        self.converters["ATAN2"] = self._convert_atan2
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
        self.converters["REVERSE"] = self._convert_reverse_v2  # REVERSE is same as REVERSE_V2
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
        self.converters["TOP_K"] = self._convert_topk_v2  # TOP_K is same as TOPK_V2
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

        return self.converters[op_type](inputs, options)

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
                    
                    # Convert from NHWC (TFLite) to NCHW (PyTorch)
                    permute_to_nchw = graph.call_function(
                        torch.permute,
                        args=(input_nodes[0], (0, 3, 1, 2))
                    )
                    
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
                    
                    # Apply convolution
                    conv_output = graph.call_module(module_name, args=(permute_to_nchw,))
                    
                    # Convert back from NCHW to NHWC
                    output_node = graph.call_function(
                        torch.permute,
                        args=(conv_output, (0, 2, 3, 1))
                    )
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
    ) -> Callable:
        """Convert TFLite DEPTHWISE_CONV_2D to PyTorch depthwise Conv2d."""
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        padding = options.get("padding", "SAME")
        depth_multiplier = options.get("depth_multiplier", 1)
        activation = options.get("fused_activation_function", "NONE")

        if padding == "SAME":
            padding_mode = "same"
        else:
            padding_mode = 0

        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for DEPTHWISE_CONV_2D."""
            # Extract weight tensor shape to determine conv parameters
            if len(operator.inputs) >= 2:
                weight_idx = operator.inputs[1]
                weight_tensor_info = subgraph.tensors[weight_idx]
                # TFLite weight format for depthwise: [1, kernel_h, kernel_w, channels]
                if len(weight_tensor_info.shape) == 4:
                    kernel_size = (weight_tensor_info.shape[1], weight_tensor_info.shape[2])
                    in_channels = weight_tensor_info.shape[3]
                    out_channels = in_channels * depth_multiplier
                    
                    has_bias = len(operator.inputs) >= 3 and operator.inputs[2] >= 0
                    
                    # Convert from NHWC (TFLite) to NCHW (PyTorch)
                    permute_to_nchw = graph.call_function(
                        torch.permute,
                        args=(input_nodes[0], (0, 3, 1, 2))
                    )
                    
                    # Create depthwise Conv2d (groups = in_channels)
                    module = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=(stride_h, stride_w),
                        padding=padding_mode,
                        groups=in_channels,  # Depthwise
                        bias=has_bias
                    )
                    
                    # Load weights
                    if weight_idx in weights:
                        weight_tensor = weights[weight_idx]
                        # Reshape from [1, h, w, c] to [c*multiplier, 1, h, w]
                        weight_tensor = weight_tensor.squeeze(0).permute(2, 0, 1).unsqueeze(1)
                        module.weight.data = weight_tensor
                    
                    # Load bias if exists
                    if has_bias:
                        bias_idx = operator.inputs[2]
                        if bias_idx in weights and module.bias is not None:
                            module.bias.data = weights[bias_idx]
                    
                    module_name = f"module_{node_counter['count']}"
                    node_counter['count'] += 1
                    parameter_dict[module_name] = module
                    conv_output = graph.call_module(module_name, args=(permute_to_nchw,))
                    
                    # Convert back from NCHW to NHWC
                    output_node = graph.call_function(
                        torch.permute,
                        args=(conv_output, (0, 2, 3, 1))
                    )
                    output_node.name = node_name
                    
                    # Handle fused activation
                    if activation != "NONE":
                        activation_module = self.get_activation_module(activation)
                        if activation_module:
                            act_name = f"activation_{node_counter['count']}"
                            node_counter['count'] += 1
                            parameter_dict[act_name] = activation_module
                            output_node = graph.call_module(act_name, args=(output_node,))
                            output_node.name = f"{node_name}_activation"
                    
                    return output_node
            
            # Fallback
            return graph.call_function(lambda x: x, args=(input_nodes[0],) if input_nodes else ())
        return build_graph

    def _convert_fully_connected(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Callable:
        """Convert TFLite FULLY_CONNECTED to PyTorch Linear."""
        activation = options.get("fused_activation_function", "NONE")
        
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for FULLY_CONNECTED."""
            # Extract weight tensor shape
            if len(operator.inputs) >= 2:
                weight_idx = operator.inputs[1]
                weight_tensor_info = subgraph.tensors[weight_idx]
                if len(weight_tensor_info.shape) == 2:
                    out_features = weight_tensor_info.shape[0]
                    in_features = weight_tensor_info.shape[1]
                    has_bias = len(operator.inputs) >= 3 and operator.inputs[2] >= 0
                    
                    module = nn.Linear(
                        in_features=in_features,
                        out_features=out_features,
                        bias=has_bias
                    )
                    
                    # Load weights
                    if weight_idx in weights:
                        module.weight.data = weights[weight_idx]
                    
                    # Load bias if exists
                    if has_bias:
                        bias_idx = operator.inputs[2]
                        if bias_idx in weights and module.bias is not None:
                            module.bias.data = weights[bias_idx]
                    
                    module_name = f"module_{node_counter['count']}"
                    node_counter['count'] += 1
                    parameter_dict[module_name] = module
                    output_node = graph.call_module(module_name, args=(input_nodes[0],))
                    output_node.name = node_name
                    
                    # Handle fused activation
                    if activation != "NONE":
                        activation_module = self.get_activation_module(activation)
                        if activation_module:
                            act_name = f"activation_{node_counter['count']}"
                            node_counter['count'] += 1
                            parameter_dict[act_name] = activation_module
                            output_node = graph.call_module(act_name, args=(output_node,))
                            output_node.name = f"{node_name}_activation"
                    
                    return output_node
            
            # Fallback
            return graph.call_function(lambda x: x, args=(input_nodes[0],) if input_nodes else ())
        return build_graph

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

    def _convert_max_pool2d(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite MAX_POOL_2D to PyTorch MaxPool2d."""
        # Default stride of 2 is typical for pooling operations in TFLite
        stride_h = options.get("stride_h", 2)
        stride_w = options.get("stride_w", 2)
        filter_height = options.get("filter_height", 2)
        filter_width = options.get("filter_width", 2)
        padding = options.get("padding", "VALID")

        # Convert padding - VALID means no padding (0), SAME means "same"
        if padding == "SAME":
            padding_mode = "same"
        elif padding == "VALID":
            padding_mode = 0
        else:
            padding_mode = 0

        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_max_pool2d."""
            # Convert from NHWC (TFLite) to NCHW (PyTorch)
            permute_to_nchw = graph.call_function(
                torch.permute,
                args=(input_nodes[0], (0, 3, 1, 2))
            )
            
            module = nn.MaxPool2d(**{
                "kernel_size": (filter_height, filter_width),
                "stride": (stride_h, stride_w),
                "padding": padding_mode,
            })
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            pool_output = graph.call_module(module_name, args=(permute_to_nchw,))
            
            # Convert back from NCHW to NHWC
            output_node = graph.call_function(
                torch.permute,
                args=(pool_output, (0, 2, 3, 1))
            )
            output_node.name = node_name
            return output_node
        return build_graph


    def _convert_avg_pool2d(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite AVERAGE_POOL_2D to PyTorch AvgPool2d."""
        stride_h = options.get("stride_h", 2)
        stride_w = options.get("stride_w", 2)
        filter_height = options.get("filter_height", 2)
        filter_width = options.get("filter_width", 2)
        padding = options.get("padding", "VALID")

        # Convert padding - VALID means no padding (0), SAME means "same"
        if padding == "SAME":
            padding_mode = "same"
        elif padding == "VALID":
            padding_mode = 0
        else:
            padding_mode = 0

        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_avg_pool2d."""
            # Convert from NHWC (TFLite) to NCHW (PyTorch)
            permute_to_nchw = graph.call_function(
                torch.permute,
                args=(input_nodes[0], (0, 3, 1, 2))
            )
            
            module = nn.AvgPool2d(**{
                "kernel_size": (filter_height, filter_width),
                "stride": (stride_h, stride_w),
                "padding": padding_mode,
            })
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            pool_output = graph.call_module(module_name, args=(permute_to_nchw,))
            
            # Convert back from NCHW to NHWC
            output_node = graph.call_function(
                torch.permute,
                args=(pool_output, (0, 2, 3, 1))
            )
            output_node.name = node_name
            return output_node
        return build_graph


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
                    # If shape is not available as constant, raise an error
                    # Flattening to (-1,) would not preserve tensor structure
                    raise ValueError(
                        f"RESHAPE operator at {node_name} requires shape tensor as constant weight, "
                        f"but shape_idx {shape_idx} not found in weights"
                    )
            else:
                # No valid inputs - should not happen in well-formed model
                if not input_nodes:
                    raise ValueError(f"RESHAPE operator at {node_name} requires at least one input tensor")
                output_node = graph.call_function(lambda x: x, args=(input_nodes[0],))
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_concatenation(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Callable:
        """Convert TFLite CONCATENATION to PyTorch cat.
        
        CONCATENATION takes multiple input tensors and concatenates them.
        The axis is specified in options. Default is 1 (feature axis for 2D tensors).
        """
        # TFLite typically concatenates along the last non-batch axis by default
        # For 2D tensors (batch, features), this is axis=1
        axis = options.get("axis", 1)
        
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

    def _convert_pad(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite PAD to PyTorch pad.
        
        PAD takes 2 inputs: input tensor and paddings tensor.
        The paddings tensor needs to be converted to a tuple.
        TFLite format: [[pad_before_dim_0, pad_after_dim_0], ...]
        PyTorch format: (left, right, top, bottom, ...) - reversed
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_pad."""
            input_tensor = input_nodes[0]
            pad_tensor = input_nodes[1]
            
            # Custom pad conversion
            def convert_pad(input_tensor, pad_tensor):
                # Convert padding format
                pad_list = []
                ndim = pad_tensor.shape[0]
                for i in range(ndim - 1, -1, -1):
                    pad_list.extend([pad_tensor[i, 0].item(), pad_tensor[i, 1].item()])
                
                # Skip leading zero paddings
                while len(pad_list) > 0 and pad_list[-2] == 0 and pad_list[-1] == 0:
                    pad_list = pad_list[:-2]
                
                return torch.nn.functional.pad(input_tensor, tuple(pad_list), mode='constant', value=0)
            
            output_node = graph.call_function(convert_pad, args=(input_tensor, pad_tensor))
            output_node.name = node_name
            return output_node
        return build_graph


    def _convert_squeeze(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SQUEEZE to PyTorch squeeze."""
        squeeze_dims = options.get("squeeze_dims", None)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_squeeze."""
            output_node = graph.call_function(torch.squeeze, args=tuple(input_nodes), kwargs={"dim": squeeze_dims})
            output_node.name = node_name
            return output_node
        return build_graph


    def _convert_expand_dims(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite EXPAND_DIMS to PyTorch unsqueeze."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_expand_dims."""
            output_node = graph.call_function(torch.unsqueeze, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph


    def _convert_slice(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SLICE to PyTorch slice indexing.
        
        TFLite SLICE: input, begin, size
        PyTorch uses tensor indexing with slices.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_slice."""
            input_tensor = input_nodes[0]
            begin = input_nodes[1]
            size = input_nodes[2]
            
            # Custom slice implementation
            def slice_impl(input_tensor, begin, size):
                # Convert tensors to lists
                begin_list = begin.tolist() if torch.is_tensor(begin) else begin
                size_list = size.tolist() if torch.is_tensor(size) else size
                
                # Build slice objects for each dimension
                slices = []
                for b, s in zip(begin_list, size_list):
                    if s == -1:
                        # -1 means slice to the end
                        slices.append(slice(b, None))
                    else:
                        slices.append(slice(b, b + s))
                
                return input_tensor[tuple(slices)]
            
            output_node = graph.call_function(slice_impl, args=(input_tensor, begin, size))
            output_node.name = node_name
            return output_node
        return build_graph


    def _convert_gather(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite GATHER to PyTorch index_select."""
        axis = options.get("axis", 0)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_gather."""
            # torch.index_select expects (input, dim, index)
            output_node = graph.call_function(torch.index_select, args=(input_nodes[0], axis, input_nodes[1]))
            output_node.name = node_name
            return output_node
        return build_graph


    def _convert_split(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SPLIT to PyTorch split."""
        num_splits = options.get("num_splits", 1)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_split."""
            output_node = graph.call_function(torch.split, args=tuple(input_nodes), kwargs={"split_size_or_sections": num_splits})
            output_node.name = node_name
            return output_node
        return build_graph


    def _convert_batch_to_space(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Callable:
        """Convert TFLite BATCH_TO_SPACE_ND to PyTorch operations."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_batch_to_space."""
            # TODO: Implement batch_to_space logic
            output_node = graph.call_function(lambda x: x, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_space_to_batch(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Callable:
        """Convert TFLite SPACE_TO_BATCH_ND to PyTorch operations."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_space_to_batch."""
            # TODO: Implement space_to_batch logic
            output_node = graph.call_function(lambda x: x, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_resize_bilinear(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Callable:
        """Convert TFLite RESIZE_BILINEAR to PyTorch interpolate.
        
        PyTorch interpolate requires size as tuple of ints.
        """
        align_corners = options.get("align_corners", False)
        
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_resize_bilinear."""
            input_tensor = input_nodes[0]
            size_tensor = input_nodes[1]
            
            # Convert size tensor to tuple
            tolist_node = graph.call_method("tolist", args=(size_tensor,))
            tuple_node = graph.call_function(tuple, args=(tolist_node,))
            
            output_node = graph.call_function(
                nn.functional.interpolate, 
                args=(input_tensor,), 
                kwargs={"size": tuple_node, "mode": "bilinear", "align_corners": align_corners}
            )
            output_node.name = node_name
            return output_node
        return build_graph

    def _convert_resize_nearest(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Callable:
        """Convert TFLite RESIZE_NEAREST_NEIGHBOR to PyTorch interpolate.
        
        PyTorch interpolate requires size as tuple of ints.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_resize_nearest."""
            input_tensor = input_nodes[0]
            size_tensor = input_nodes[1]
            
            # Convert size tensor to tuple
            tolist_node = graph.call_method("tolist", args=(size_tensor,))
            tuple_node = graph.call_function(tuple, args=(tolist_node,))
            
            output_node = graph.call_function(
                nn.functional.interpolate, 
                args=(input_tensor,), 
                kwargs={"size": tuple_node, "mode": "nearest"}
            )
            output_node.name = node_name
            return output_node
        return build_graph

    # Additional Arithmetic & Math Operations
    def _convert_abs(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite ABS to PyTorch abs."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_abs."""
            output_node = graph.call_function(torch.abs, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_add_n(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite ADD_N to PyTorch sum of tensors.
        
        ADD_N sums multiple tensors element-wise.
        We use torch.stack to combine tensors, then sum along the new dimension.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_add_n."""
            # Stack all input tensors along a new dimension, then sum
            stacked = graph.call_function(torch.stack, args=(input_nodes,))
            output_node = graph.call_function(torch.sum, args=(stacked,), kwargs={"dim": 0})
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_ceil(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite CEIL to PyTorch ceil."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_ceil."""
            output_node = graph.call_function(torch.ceil, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_cos(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite COS to PyTorch cos."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_cos."""
            output_node = graph.call_function(torch.cos, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_exp(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite EXP to PyTorch exp."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_exp."""
            output_node = graph.call_function(torch.exp, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_floor(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite FLOOR to PyTorch floor."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_floor."""
            output_node = graph.call_function(torch.floor, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_floor_div(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite FLOOR_DIV to PyTorch floor_divide."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_floor_div."""
            output_node = graph.call_function(torch.floor_divide, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_floor_mod(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite FLOOR_MOD to PyTorch fmod."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_floor_mod."""
            output_node = graph.call_function(torch.fmod, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_log(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LOG to PyTorch log."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_log."""
            output_node = graph.call_function(torch.log, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_maximum(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite MAXIMUM to PyTorch maximum."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_maximum."""
            output_node = graph.call_function(torch.maximum, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_minimum(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite MINIMUM to PyTorch minimum."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_minimum."""
            output_node = graph.call_function(torch.minimum, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_neg(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite NEG to PyTorch neg."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_neg."""
            output_node = graph.call_function(torch.neg, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_pow(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite POW to PyTorch pow."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_pow."""
            output_node = graph.call_function(torch.pow, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_atan2(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite ATAN2 to PyTorch atan2."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_atan2."""
            # atan2(y, x) requires exactly two inputs
            if len(input_nodes) < 2:
                raise ValueError(f"ATAN2 requires 2 inputs, got {len(input_nodes)}")
            output_node = graph.call_function(torch.atan2, args=(input_nodes[0], input_nodes[1]))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_rsqrt(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite RSQRT to PyTorch rsqrt."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_rsqrt."""
            output_node = graph.call_function(torch.rsqrt, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_sin(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SIN to PyTorch sin."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_sin."""
            output_node = graph.call_function(torch.sin, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_square(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SQUARE to PyTorch square."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_square."""
            output_node = graph.call_function(torch.square, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_squared_difference(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SQUARED_DIFFERENCE to PyTorch operations.
        
        squared_difference(x, y) = (x - y) ** 2
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_squared_difference."""
            x = input_nodes[0]
            y = input_nodes[1]
            
            # (x - y) ** 2
            diff_node = graph.call_function(torch.sub, args=(x, y))
            output_node = graph.call_function(torch.pow, args=(diff_node, 2))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_sqrt(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SQRT to PyTorch sqrt."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_sqrt."""
            output_node = graph.call_function(torch.sqrt, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Additional Convolution Operations
    def _convert_conv3d(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite CONV_3D to PyTorch Conv3d."""
        stride = options.get("stride", [1, 1, 1])
        padding = options.get("padding", "VALID")
        dilation = options.get("dilation", [1, 1, 1])
        activation = options.get("fused_activation_function", "NONE")
        
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_conv3d."""
            # Extract weight tensor shape to determine conv parameters
            if len(operator.inputs) >= 2:
                weight_idx = operator.inputs[1]
                weight_tensor_info = subgraph.tensors[weight_idx]
                # TFLite weight format: [kernel_d, kernel_h, kernel_w, in_channels, out_channels]
                if len(weight_tensor_info.shape) == 5:
                    kernel_size = (weight_tensor_info.shape[0], weight_tensor_info.shape[1], weight_tensor_info.shape[2])
                    in_channels = weight_tensor_info.shape[3]
                    out_channels = weight_tensor_info.shape[4]
                    
                    # Check if bias exists
                    has_bias = len(operator.inputs) >= 3 and operator.inputs[2] >= 0
                    
                    # Convert from NDHWC (TFLite) to NCDHW (PyTorch)
                    permute_to_ncdhw = graph.call_function(
                        torch.permute,
                        args=(input_nodes[0], (0, 4, 1, 2, 3))
                    )
                    
                    # Create Conv3d module
                    params = {
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "kernel_size": kernel_size,
                        "stride": tuple(stride) if isinstance(stride, list) else stride,
                        "padding": 0 if padding == "VALID" else "same",
                        "dilation": tuple(dilation) if isinstance(dilation, list) else dilation,
                        "bias": has_bias
                    }
                    module = nn.Conv3d(**params)
                    
                    # Load weights
                    if weight_idx in weights:
                        weight_tensor = weights[weight_idx]
                        # Convert from TFLite format to PyTorch format
                        # TFLite: [kernel_d, kernel_h, kernel_w, in_channels, out_channels]
                        # PyTorch: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
                        weight_tensor = weight_tensor.permute(4, 3, 0, 1, 2)
                        module.weight.data = weight_tensor
                    
                    # Load bias if it exists
                    if has_bias:
                        bias_idx = operator.inputs[2]
                        if bias_idx in weights:
                            module.bias.data = weights[bias_idx]
                    
                    module_name = f"module_{node_counter['count']}"
                    node_counter['count'] += 1
                    parameter_dict[module_name] = module
                    
                    output_node = graph.call_module(module_name, args=(permute_to_ncdhw,))
                    
                    # Convert back from NCDHW to NDHWC
                    output_node = graph.call_function(
                        torch.permute,
                        args=(output_node, (0, 2, 3, 4, 1))
                    )
                    
                    # Handle fused activation
                    if activation != "NONE":
                        activation_module = self.get_activation_module(activation)
                        if activation_module:
                            act_name = f"activation_{node_counter['count']}"
                            node_counter['count'] += 1
                            parameter_dict[act_name] = activation_module
                            output_node = graph.call_module(act_name, args=(output_node,))
                            output_node.name = f"{node_name}_activation"
                    else:
                        output_node.name = node_name
            else:
                # Fallback if weight info is not available
                output_node = input_nodes[0] if input_nodes else graph.call_function(lambda: None, args=())
                output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_transpose_conv(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite TRANSPOSE_CONV to PyTorch ConvTranspose2d."""
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        padding = options.get("padding", "SAME")
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_transpose_conv."""
            module = nn.ConvTranspose2d(**{
                "stride": (stride_h, stride_w),
                "padding": 0 if padding == "VALID" else "same",
            })
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            # Handle fused activation
            activation = options.get("fused_activation_function", "NONE")
            if activation != "NONE":
                activation_module = self.get_activation_module(activation)
                if activation_module:
                    act_name = f"activation_{node_counter['count']}"
                    node_counter['count'] += 1
                    parameter_dict[act_name] = activation_module
                    output_node = graph.call_module(act_name, args=(output_node,))
                    output_node.name = f"{node_name}_activation"
            return output_node
        return build_graph

    
    def _convert_batch_matmul(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite BATCH_MATMUL to PyTorch bmm."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_batch_matmul."""
            output_node = graph.call_function(torch.bmm, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Additional Activation Functions
    def _convert_elu(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite ELU to PyTorch ELU."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_elu."""
            module = nn.ELU()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_gelu(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite GELU to PyTorch GELU."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_gelu."""
            module = nn.GELU()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_hard_swish(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite HARD_SWISH to PyTorch Hardswish."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_hard_swish."""
            module = nn.Hardswish()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_leaky_relu(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LEAKY_RELU to PyTorch LeakyReLU."""
        alpha = options.get("alpha", 0.01)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_leaky_relu."""
            module = nn.LeakyReLU(**{"negative_slope": alpha})
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_log_softmax(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LOG_SOFTMAX to PyTorch LogSoftmax."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_log_softmax."""
            module = nn.LogSoftmax(**{"dim": -1})
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_prelu(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite PRELU to PyTorch PReLU.
        
        PReLU has 2 inputs: input tensor and alpha (slope) tensor.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_prelu."""
            # Get alpha from weights or inputs
            alpha_data = None
            if len(operator.inputs) >= 2:
                alpha_idx = operator.inputs[1]
                if alpha_idx in weights:
                    alpha_data = weights[alpha_idx]
            
            if alpha_data is not None:
                # Create PReLU with the alpha values
                num_parameters = alpha_data.numel()
                module = nn.PReLU(num_parameters=num_parameters)
                module.weight.data = alpha_data.flatten()
            else:
                # Default PReLU
                module = nn.PReLU()
            
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Normalization Operations
    def _convert_l2_normalization(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite L2_NORMALIZATION to PyTorch normalize."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_l2_normalization."""
            output_node = graph.call_function(torch.nn.functional.normalize, args=tuple(input_nodes), kwargs={"p": 2, "dim": -1})
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_local_response_normalization(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LOCAL_RESPONSE_NORMALIZATION to PyTorch LocalResponseNorm."""
        size = options.get("radius", 5) * 2 + 1
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_local_response_normalization."""
            module = nn.LocalResponseNorm(**{"size": size})
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Additional Reduction Operations
    def _convert_reduce_max(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite REDUCE_MAX to PyTorch max.
        
        REDUCE_MAX takes 2 inputs: input tensor and reduction_indices tensor.
        """
        keep_dims = options.get("keep_dims", True)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_reduce_max."""
            # Extract reduction axes from the second input
            if len(operator.inputs) >= 2:
                axes_idx = operator.inputs[1]
                if axes_idx in weights:
                    axes_tensor = weights[axes_idx]
                    try:
                        if hasattr(axes_tensor, 'numpy'):
                            axes = axes_tensor.numpy().tolist()
                        elif hasattr(axes_tensor, 'tolist'):
                            axes = axes_tensor.tolist()
                        else:
                            axes = list(axes_tensor)
                    except (AttributeError, TypeError) as e:
                        raise ValueError(f"Cannot convert axes tensor to list: {e}")
                    
                    if not isinstance(axes, list):
                        axes = [axes]
                    
                    # PyTorch amax supports multiple dimensions
                    output_node = graph.call_function(
                        torch.amax,
                        args=(input_nodes[0],),
                        kwargs={"dim": axes, "keepdim": keep_dims}
                    )
                    output_node.name = node_name
                    return output_node
            
            # Fallback: max over all dimensions
            output_node = graph.call_function(torch.max, args=(input_nodes[0],))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_reduce_min(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite REDUCE_MIN to PyTorch min.
        
        REDUCE_MIN takes 2 inputs: input tensor and reduction_indices tensor.
        """
        keep_dims = options.get("keep_dims", True)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_reduce_min."""
            # Extract reduction axes from the second input
            if len(operator.inputs) >= 2:
                axes_idx = operator.inputs[1]
                if axes_idx in weights:
                    axes_tensor = weights[axes_idx]
                    try:
                        if hasattr(axes_tensor, 'numpy'):
                            axes = axes_tensor.numpy().tolist()
                        elif hasattr(axes_tensor, 'tolist'):
                            axes = axes_tensor.tolist()
                        else:
                            axes = list(axes_tensor)
                    except (AttributeError, TypeError) as e:
                        raise ValueError(f"Cannot convert axes tensor to list: {e}")
                    
                    if not isinstance(axes, list):
                        axes = [axes]
                    
                    # PyTorch amin supports multiple dimensions
                    output_node = graph.call_function(
                        torch.amin,
                        args=(input_nodes[0],),
                        kwargs={"dim": axes, "keepdim": keep_dims}
                    )
                    output_node.name = node_name
                    return output_node
            
            # Fallback: min over all dimensions
            output_node = graph.call_function(torch.min, args=(input_nodes[0],))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_reduce_prod(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite REDUCE_PROD to PyTorch prod."""
        keep_dims = options.get("keep_dims", True)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_reduce_prod."""
            # Extract reduction axes from the second input
            if len(operator.inputs) >= 2:
                axes_idx = operator.inputs[1]
                if axes_idx in weights:
                    axes_tensor = weights[axes_idx]
                    try:
                        if hasattr(axes_tensor, 'numpy'):
                            axes = axes_tensor.numpy().tolist()
                        elif hasattr(axes_tensor, 'tolist'):
                            axes = axes_tensor.tolist()
                        else:
                            axes = list(axes_tensor)
                    except (AttributeError, TypeError) as e:
                        raise ValueError(f"Cannot convert axes tensor to list: {e}")
                    
                    if not isinstance(axes, list):
                        axes = [axes]
                    
                    # For multiple axes, need to iterate
                    if len(axes) == 1:
                        output_node = graph.call_function(
                            torch.prod,
                            args=(input_nodes[0],),
                            kwargs={"dim": axes[0], "keepdim": keep_dims}
                        )
                    else:
                        # Reduce over multiple dimensions sequentially
                        output_node = input_nodes[0]
                        for axis in sorted(axes, reverse=True):
                            output_node = graph.call_function(
                                torch.prod,
                                args=(output_node,),
                                kwargs={"dim": axis, "keepdim": keep_dims}
                            )
                    output_node.name = node_name
                    return output_node
            
            # Fallback
            output_node = graph.call_function(torch.prod, args=(input_nodes[0],), kwargs={"keepdim": keep_dims})
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_reduce_any(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite REDUCE_ANY to PyTorch any."""
        keep_dims = options.get("keep_dims", True)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_reduce_any."""
            # Extract reduction axes from the second input
            if len(operator.inputs) >= 2:
                axes_idx = operator.inputs[1]
                if axes_idx in weights:
                    axes_tensor = weights[axes_idx]
                    try:
                        if hasattr(axes_tensor, 'numpy'):
                            axes = axes_tensor.numpy().tolist()
                        elif hasattr(axes_tensor, 'tolist'):
                            axes = axes_tensor.tolist()
                        else:
                            axes = list(axes_tensor)
                    except (AttributeError, TypeError) as e:
                        raise ValueError(f"Cannot convert axes tensor to list: {e}")
                    
                    if not isinstance(axes, list):
                        axes = [axes]
                    
                    # For single axis
                    if len(axes) == 1:
                        output_node = graph.call_function(
                            torch.any,
                            args=(input_nodes[0],),
                            kwargs={"dim": axes[0], "keepdim": keep_dims}
                        )
                    else:
                        # Multiple axes: reduce sequentially
                        output_node = input_nodes[0]
                        for axis in sorted(axes, reverse=True):
                            output_node = graph.call_function(
                                torch.any,
                                args=(output_node,),
                                kwargs={"dim": axis, "keepdim": keep_dims}
                            )
                    output_node.name = node_name
                    return output_node
            
            # Fallback
            output_node = graph.call_function(torch.any, args=(input_nodes[0],), kwargs={"keepdim": keep_dims})
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_sum(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SUM to PyTorch sum.
        
        SUM takes 2 inputs: input tensor and reduction_indices tensor.
        """
        # Default to True for compatibility with typical TFLite reduction operations
        keep_dims = options.get("keep_dims", True)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_sum."""
            # Extract reduction axes from the second input (reduction_indices)
            if len(operator.inputs) >= 2:
                axes_idx = operator.inputs[1]
                if axes_idx in weights:
                    axes_tensor = weights[axes_idx]
                    # Convert to Python list of ints
                    try:
                        if hasattr(axes_tensor, 'numpy'):
                            axes = axes_tensor.numpy().tolist()
                        elif hasattr(axes_tensor, 'tolist'):
                            axes = axes_tensor.tolist()
                        else:
                            axes = list(axes_tensor)
                    except (AttributeError, TypeError) as e:
                        raise ValueError(f"Cannot convert axes tensor to list: {e}")
                    
                    if not isinstance(axes, list):
                        axes = [axes]
                    
                    # Perform the sum reduction
                    output_node = graph.call_function(
                        torch.sum,
                        args=(input_nodes[0],),
                        kwargs={"dim": axes, "keepdim": keep_dims}
                    )
                    output_node.name = node_name
                    return output_node
            
            # Fallback: sum over all dimensions
            output_node = graph.call_function(torch.sum, args=(input_nodes[0],))
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Additional Shape & Tensor Manipulation
    def _convert_broadcast_args(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite BROADCAST_ARGS to PyTorch broadcast_shapes.
        
        PyTorch broadcast_shapes requires shapes as tuples, not tensors.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_broadcast_args."""
            shape1_tensor = input_nodes[0]
            shape2_tensor = input_nodes[1]
            
            # Convert shape tensors to tuples
            shape1_list = graph.call_method("tolist", args=(shape1_tensor,))
            shape1_tuple = graph.call_function(tuple, args=(shape1_list,))
            
            shape2_list = graph.call_method("tolist", args=(shape2_tensor,))
            shape2_tuple = graph.call_function(tuple, args=(shape2_list,))
            
            output_node = graph.call_function(torch.broadcast_shapes, args=(shape1_tuple, shape2_tuple))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_broadcast_to(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite BROADCAST_TO to PyTorch broadcast_to.
        
        PyTorch broadcast_to requires size as tuple of ints, not a tensor.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_broadcast_to."""
            input_tensor = input_nodes[0]
            shape_tensor = input_nodes[1]
            
            # Convert shape tensor to tuple
            tolist_node = graph.call_method("tolist", args=(shape_tensor,))
            tuple_node = graph.call_function(tuple, args=(tolist_node,))
            
            output_node = graph.call_function(torch.broadcast_to, args=(input_tensor, tuple_node))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_depth_to_space(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite DEPTH_TO_SPACE to PyTorch pixel_shuffle."""
        block_size = options.get("block_size", 2)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_depth_to_space."""
            # Convert from NHWC (TFLite) to NCHW (PyTorch)
            permute_to_nchw = graph.call_function(
                torch.permute,
                args=(input_nodes[0], (0, 3, 1, 2))
            )
            
            module = nn.PixelShuffle(**{"upscale_factor": block_size})
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            shuffle_output = graph.call_module(module_name, args=(permute_to_nchw,))
            
            # Convert back from NCHW to NHWC
            output_node = graph.call_function(
                torch.permute,
                args=(shuffle_output, (0, 2, 3, 1))
            )
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_fill(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite FILL to PyTorch full."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_fill."""
            output_node = graph.call_function(torch.full, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_gather_nd(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite GATHER_ND to PyTorch operations.
        
        gather_nd gathers slices from params according to indices.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_gather_nd."""
            params = input_nodes[0]
            indices = input_nodes[1]
            
            # Custom gather_nd implementation
            def gather_nd_impl(params, indices):
                # indices shape: [..., N] where N is the number of indexed dimensions
                # Convert multi-dimensional indices to access params
                indices_list = indices.unbind(-1)
                return params[tuple(indices_list)]
            
            output_node = graph.call_function(gather_nd_impl, args=(params, indices))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_mirror_pad(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite MIRROR_PAD to PyTorch pad with reflect mode.
        
        PyTorch pad requires pad to be a tuple of ints, not a tensor.
        TFLite format: [[pad_before_dim_0, pad_after_dim_0], [pad_before_dim_1, pad_after_dim_1], ...]
        PyTorch format: (left, right, top, bottom, front, back, ...) - reversed and flattened
        """
        mode = options.get("mode", "REFLECT")
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_mirror_pad."""
            input_tensor = input_nodes[0]
            pad_tensor = input_nodes[1]
            
            # Custom pad conversion: TFLite to PyTorch format
            def convert_pad_tflite_to_torch(input_tensor, pad_tensor):
                # pad_tensor shape: [ndim, 2]
                # Convert to PyTorch format: reverse dimension order and flatten
                # PyTorch pads from last dimension to first
                pad_list = []
                ndim = pad_tensor.shape[0]
                for i in range(ndim - 1, -1, -1):
                    pad_list.extend([pad_tensor[i, 0].item(), pad_tensor[i, 1].item()])
                
                # PyTorch pad only supports padding last 2 or 3 dimensions for 4D input
                # Skip leading zero paddings
                while len(pad_list) > 0 and pad_list[-2] == 0 and pad_list[-1] == 0:
                    pad_list = pad_list[:-2]
                
                return torch.nn.functional.pad(input_tensor, tuple(pad_list), mode='reflect')
            
            output_node = graph.call_function(convert_pad_tflite_to_torch, args=(input_tensor, pad_tensor))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_pack(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite PACK to PyTorch stack.
        
        PACK takes multiple input tensors and stacks them along a new dimension.
        PyTorch stack expects tensors as a sequence (tuple/list).
        """
        axis = options.get("axis", 0)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_pack."""
            # TODO: Implement custom operator logic
            if input_nodes:
                output_node = graph.call_function(lambda x: x, args=(input_nodes[0],))
            else:
                output_node = graph.call_function(lambda: None, args=())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_padv2(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite PADV2 to PyTorch pad.
        
        PADV2 takes 3 inputs: input tensor, paddings tensor, and constant_values.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_padv2."""
            input_tensor = input_nodes[0]
            pad_tensor = input_nodes[1]
            constant_value = input_nodes[2] if len(input_nodes) > 2 else None
            
            # Custom pad conversion
            def convert_padv2(input_tensor, pad_tensor, constant_value=0):
                # Convert padding format
                pad_list = []
                ndim = pad_tensor.shape[0]
                for i in range(ndim - 1, -1, -1):
                    pad_list.extend([pad_tensor[i, 0].item(), pad_tensor[i, 1].item()])
                
                # Skip leading zero paddings
                while len(pad_list) > 0 and pad_list[-2] == 0 and pad_list[-1] == 0:
                    pad_list = pad_list[:-2]
                
                # Get constant value if it's a tensor
                if torch.is_tensor(constant_value):
                    constant_value = constant_value.item()
                
                return torch.nn.functional.pad(input_tensor, tuple(pad_list), mode='constant', value=constant_value)
            
            if constant_value is not None:
                output_node = graph.call_function(convert_padv2, args=(input_tensor, pad_tensor, constant_value))
            else:
                output_node = graph.call_function(convert_padv2, args=(input_tensor, pad_tensor))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_range(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite RANGE to PyTorch arange."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_range."""
            output_node = graph.call_function(torch.arange, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_reverse_v2(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite REVERSE_V2 to PyTorch flip.
        
        PyTorch flip requires dims as tuple of ints, not a tensor.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_reverse_v2."""
            input_tensor = input_nodes[0]
            axis_tensor = input_nodes[1]
            
            # Convert axis tensor to tuple
            tolist_node = graph.call_method("tolist", args=(axis_tensor,))
            tuple_node = graph.call_function(tuple, args=(tolist_node,))
            
            output_node = graph.call_function(torch.flip, args=(input_tensor, tuple_node))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_reverse_sequence(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite REVERSE_SEQUENCE to PyTorch operations.
        
        Reverses variable length slices along a specified dimension.
        """
        seq_axis = options.get("seq_dim", 1)
        batch_axis = options.get("batch_dim", 0)
        
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_reverse_sequence."""
            x = input_nodes[0]
            seq_lengths = input_nodes[1]
            
            # Custom reverse_sequence implementation
            def reverse_sequence_impl(x, seq_lengths, seq_axis=seq_axis, batch_axis=batch_axis):
                output = x.clone()
                batch_size = x.shape[batch_axis]
                
                for i in range(batch_size):
                    seq_len = seq_lengths[i].item()
                    # Get the slice for this batch element
                    if batch_axis == 0:
                        if seq_axis == 1:
                            # Reverse the first seq_len elements along axis 1
                            output[i, :seq_len] = torch.flip(x[i, :seq_len], dims=[0])
                        else:
                            indices = [slice(None)] * x.ndim
                            indices[batch_axis] = i
                            indices[seq_axis] = slice(0, seq_len)
                            output[tuple(indices)] = torch.flip(x[tuple(indices)], dims=[seq_axis])
                    else:
                        indices = [slice(None)] * x.ndim
                        indices[batch_axis] = i
                        indices[seq_axis] = slice(0, seq_len)
                        output[tuple(indices)] = torch.flip(x[tuple(indices)], dims=[seq_axis])
                
                return output
            
            output_node = graph.call_function(reverse_sequence_impl, args=(x, seq_lengths))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_scatter_nd(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SCATTER_ND to PyTorch operations.
        
        TFLite SCATTER_ND: indices, updates, shape
        Creates a tensor of given shape and scatters updates at indices.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_scatter_nd."""
            indices = input_nodes[0]
            updates = input_nodes[1]
            shape = input_nodes[2]
            
            # Custom scatter_nd implementation
            def scatter_nd_impl(indices, updates, shape):
                # Convert shape to list if it's a tensor
                if torch.is_tensor(shape):
                    shape_list = shape.tolist()
                else:
                    shape_list = list(shape)
                
                # Create output tensor filled with zeros
                output = torch.zeros(shape_list, dtype=updates.dtype, device=updates.device)
                
                # Scatter updates at indices
                # indices shape: [..., N] where N is the rank of the indexed dimensions
                # Convert multi-dimensional indices to linear indices for advanced indexing
                indices_list = indices.unbind(-1)
                output[tuple(indices_list)] = updates
                
                return output
            
            output_node = graph.call_function(scatter_nd_impl, args=(indices, updates, shape))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_shape(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SHAPE to PyTorch shape property."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_shape."""
            # TODO: Implement custom operator logic
            if input_nodes:
                output_node = graph.call_function(lambda x: x, args=(input_nodes[0],))
            else:
                output_node = graph.call_function(lambda: None, args=())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_space_to_depth(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SPACE_TO_DEPTH to PyTorch pixel_unshuffle."""
        block_size = options.get("block_size", 2)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_space_to_depth."""
            # Convert from NHWC (TFLite) to NCHW (PyTorch)
            permute_to_nchw = graph.call_function(
                torch.permute,
                args=(input_nodes[0], (0, 3, 1, 2))
            )
            
            module = nn.PixelUnshuffle(**{"downscale_factor": block_size})
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            unshuffle_output = graph.call_module(module_name, args=(permute_to_nchw,))
            
            # Convert back from NCHW to NHWC
            output_node = graph.call_function(
                torch.permute,
                args=(unshuffle_output, (0, 2, 3, 1))
            )
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_sparse_to_dense(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SPARSE_TO_DENSE to PyTorch operations."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_sparse_to_dense."""
            # TODO: Implement custom operator logic
            if input_nodes:
                output_node = graph.call_function(lambda x: x, args=(input_nodes[0],))
            else:
                output_node = graph.call_function(lambda: None, args=())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_split_v(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SPLIT_V to PyTorch split.
        
        TFLite SPLIT_V: input, size_splits, axis
        PyTorch split requires split_sizes as tuple/list of ints.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_split_v."""
            input_tensor = input_nodes[0]
            size_splits = input_nodes[1]
            axis = input_nodes[2]
            
            # Convert size_splits tensor to tuple
            tolist_node = graph.call_method("tolist", args=(size_splits,))
            
            output_node = graph.call_function(torch.split, args=(input_tensor, tolist_node, axis))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_strided_slice(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite STRIDED_SLICE to PyTorch slice.
        
        TFLite STRIDED_SLICE: input, begin, end, strides
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_strided_slice."""
            input_tensor = input_nodes[0]
            begin = input_nodes[1]
            end = input_nodes[2]
            strides = input_nodes[3] if len(input_nodes) > 3 else None
            
            # Custom strided slice implementation
            def strided_slice_impl(input_tensor, begin, end, strides=None):
                # Convert tensors to lists
                begin_list = begin.tolist() if torch.is_tensor(begin) else begin
                end_list = end.tolist() if torch.is_tensor(end) else end
                stride_list = strides.tolist() if strides is not None and torch.is_tensor(strides) else ([1] * len(begin_list) if strides is None else strides)
                
                # Build slice objects for each dimension
                slices = []
                for b, e, s in zip(begin_list, end_list, stride_list):
                    slices.append(slice(b, e, s))
                
                return input_tensor[tuple(slices)]
            
            if strides is not None:
                output_node = graph.call_function(strided_slice_impl, args=(input_tensor, begin, end, strides))
            else:
                output_node = graph.call_function(strided_slice_impl, args=(input_tensor, begin, end))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_tile(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite TILE to PyTorch tile.
        
        PyTorch tile requires dims as tuple of ints, not a tensor.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_tile."""
            input_tensor = input_nodes[0]
            multiples_tensor = input_nodes[1]
            
            # Convert multiples tensor to tuple
            tolist_node = graph.call_method("tolist", args=(multiples_tensor,))
            tuple_node = graph.call_function(tuple, args=(tolist_node,))
            
            output_node = graph.call_function(torch.tile, args=(input_tensor, tuple_node))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_topk_v2(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite TOPK_V2 to PyTorch topk."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_topk_v2."""
            # TFLite TOPK_V2 has 2 inputs: input tensor and k (as a constant)
            # We need to get k from the second input if it's a constant
            k_value = 1
            if len(operator.inputs) >= 2:
                k_idx = operator.inputs[1]
                if k_idx >= 0 and k_idx in weights:
                    k_tensor = weights[k_idx]
                    k_value = int(k_tensor.item())
            
            # topk returns (values, indices), we need to extract values with getitem
            topk_output = graph.call_function(torch.topk, args=(input_nodes[0],), kwargs={"k": k_value})
            # Get the values (first element of the tuple) using operator.getitem
            import operator as op
            output_node = graph.call_function(op.getitem, args=(topk_output, 0))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_unpack(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite UNPACK to PyTorch unbind."""
        axis = options.get("axis", 0)
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_unpack."""
            output_node = graph.call_function(torch.unbind, args=tuple(input_nodes), kwargs={"dim": axis})
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_unique(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite UNIQUE to PyTorch unique."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_unique."""
            output_node = graph.call_function(torch.unique, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_where(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite WHERE to PyTorch where."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_where."""
            output_node = graph.call_function(torch.where, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_zeros_like(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite ZEROS_LIKE to PyTorch zeros_like."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_zeros_like."""
            output_node = graph.call_function(torch.zeros_like, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Comparison Operations
    def _convert_equal(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite EQUAL to PyTorch eq."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_equal."""
            output_node = graph.call_function(torch.eq, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_greater(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite GREATER to PyTorch gt."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_greater."""
            output_node = graph.call_function(torch.gt, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_greater_equal(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite GREATER_EQUAL to PyTorch ge."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_greater_equal."""
            output_node = graph.call_function(torch.ge, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_less(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LESS to PyTorch lt."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_less."""
            output_node = graph.call_function(torch.lt, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_less_equal(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LESS_EQUAL to PyTorch le."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_less_equal."""
            output_node = graph.call_function(torch.le, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_not_equal(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite NOT_EQUAL to PyTorch ne."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_not_equal."""
            output_node = graph.call_function(torch.ne, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Logical Operations
    def _convert_logical_and(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LOGICAL_AND to PyTorch logical_and."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_logical_and."""
            output_node = graph.call_function(torch.logical_and, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_logical_not(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LOGICAL_NOT to PyTorch logical_not."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_logical_not."""
            output_node = graph.call_function(torch.logical_not, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_logical_or(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LOGICAL_OR to PyTorch logical_or."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_logical_or."""
            output_node = graph.call_function(torch.logical_or, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Selection Operations
    def _convert_arg_max(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite ARG_MAX to PyTorch argmax."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_arg_max."""
            output_node = graph.call_function(torch.argmax, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_arg_min(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite ARG_MIN to PyTorch argmin."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_arg_min."""
            output_node = graph.call_function(torch.argmin, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_one_hot(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite ONE_HOT to PyTorch one_hot.
        
        TFLite ONE_HOT takes 4 inputs: indices, depth, on_value, off_value
        PyTorch one_hot only takes indices and num_classes.
        We use the first two inputs and scale/shift the result if needed.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_one_hot."""
            # PyTorch one_hot only takes (indices, num_classes)
            # and requires indices to be of type long
            indices = input_nodes[0]
            indices_long = graph.call_method("long", args=(indices,))
            depth = input_nodes[1] if len(input_nodes) > 1 else None
            
            if depth is not None:
                # Call one_hot with indices and depth
                output_node = graph.call_function(torch.nn.functional.one_hot, args=(indices_long, depth))
            else:
                output_node = graph.call_function(torch.nn.functional.one_hot, args=(indices_long,))
            
            # Handle on_value and off_value if provided
            if len(input_nodes) > 2:
                on_value = input_nodes[2]
                off_value = input_nodes[3] if len(input_nodes) > 3 else None
                
                if off_value is not None:
                    # Scale: multiply by (on_value - off_value) and add off_value
                    # result = one_hot * (on_value - off_value) + off_value
                    diff_node = graph.call_function(torch.sub, args=(on_value, off_value))
                    scaled_node = graph.call_function(torch.mul, args=(output_node, diff_node))
                    output_node = graph.call_function(torch.add, args=(scaled_node, off_value))
            
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_select(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SELECT to PyTorch where."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_select."""
            output_node = graph.call_function(torch.where, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_select_v2(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SELECT_V2 to PyTorch where."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_select_v2."""
            output_node = graph.call_function(torch.where, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Recurrent Neural Network Operations
    def _convert_lstm(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite LSTM to PyTorch LSTM."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_lstm."""
            module = nn.LSTM()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_bidirectional_sequence_lstm(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite BIDIRECTIONAL_SEQUENCE_LSTM to PyTorch LSTM."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_bidirectional_sequence_lstm."""
            module = nn.LSTM(**{"bidirectional": True})
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_unidirectional_sequence_lstm(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite UNIDIRECTIONAL_SEQUENCE_LSTM to PyTorch LSTM."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_unidirectional_sequence_lstm."""
            module = nn.LSTM()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_rnn(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite RNN to PyTorch RNN."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_rnn."""
            module = nn.RNN()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_bidirectional_sequence_rnn(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite BIDIRECTIONAL_SEQUENCE_RNN to PyTorch RNN."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_bidirectional_sequence_rnn."""
            module = nn.RNN(**{"bidirectional": True})
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_unidirectional_sequence_rnn(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite UNIDIRECTIONAL_SEQUENCE_RNN to PyTorch RNN."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_unidirectional_sequence_rnn."""
            module = nn.RNN()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Quantization Operations
    def _convert_quantize(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite QUANTIZE to PyTorch quantize_per_tensor."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_quantize."""
            output_node = graph.call_function(torch.quantize_per_tensor, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_dequantize(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite DEQUANTIZE to PyTorch dequantize."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_dequantize."""
            # TODO: Implement custom operator logic
            if input_nodes:
                output_node = graph.call_function(lambda x: x, args=(input_nodes[0],))
            else:
                output_node = graph.call_function(lambda: None, args=())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_fake_quant(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite FAKE_QUANT to PyTorch fake_quantize_per_tensor_affine."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_fake_quant."""
            output_node = graph.call_function(torch.fake_quantize_per_tensor_affine, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Type Conversion
    def _convert_cast(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite CAST to PyTorch to/type conversion."""
        # Get output dtype from options or infer from output tensor
        out_data_type = options.get("out_data_type", None)
        
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_cast."""
            # Map numpy dtype strings to PyTorch dtypes
            dtype_str_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'int32': torch.int32,
                'uint8': torch.uint8,
                'int64': torch.int64,
                'bool': torch.bool,
                'int16': torch.int16,
                'complex64': torch.complex64,
                'int8': torch.int8,
                'complex128': torch.complex128,
            }
            
            # Get target dtype from output tensor
            target_dtype = torch.float32  # default
            if len(operator.outputs) > 0:
                output_idx = operator.outputs[0]
                output_tensor = subgraph.tensors[output_idx]
                target_dtype = dtype_str_map.get(output_tensor.dtype, torch.float32)
            
            # Cast the input
            output_node = graph.call_method("to", args=(input_nodes[0], target_dtype))
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Embedding & Lookup
    def _convert_embedding_lookup(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite EMBEDDING_LOOKUP to PyTorch Embedding."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_embedding_lookup."""
            module = nn.Embedding()
            module_name = f"module_{node_counter['count']}"
            node_counter['count'] += 1
            parameter_dict[module_name] = module
            output_node = graph.call_module(module_name, args=(input_nodes[0],) if input_nodes else ())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_hashtable_lookup(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite HASHTABLE_LOOKUP to custom implementation."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_hashtable_lookup."""
            # TODO: Implement custom operator logic
            if input_nodes:
                output_node = graph.call_function(lambda x: x, args=(input_nodes[0],))
            else:
                output_node = graph.call_function(lambda: None, args=())
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Custom & Advanced Operations
    def _convert_custom(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite CUSTOM operation."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_custom."""
            # TODO: Implement custom operator logic
            if input_nodes:
                output_node = graph.call_function(lambda x: x, args=(input_nodes[0],))
            else:
                output_node = graph.call_function(lambda: None, args=())
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_cumsum(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite CUMSUM to PyTorch cumsum."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_cumsum."""
            output_node = graph.call_function(torch.cumsum, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_matrix_diag(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite MATRIX_DIAG to PyTorch diag."""
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_matrix_diag."""
            output_node = graph.call_function(torch.diag, args=tuple(input_nodes))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_matrix_set_diag(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite MATRIX_SET_DIAG to custom implementation.
        
        Sets the diagonal of a matrix to a given vector.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_matrix_set_diag."""
            matrix = input_nodes[0]
            diagonal = input_nodes[1]
            
            # Custom matrix_set_diag implementation
            def matrix_set_diag_impl(matrix, diagonal):
                output = matrix.clone()
                # Set diagonal using torch's diagonal functionality
                output.diagonal().copy_(diagonal)
                return output
            
            output_node = graph.call_function(matrix_set_diag_impl, args=(matrix, diagonal))
            output_node.name = node_name
            return output_node
        return build_graph

    
    def _convert_segment_sum(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """Convert TFLite SEGMENT_SUM to PyTorch segment operations.
        
        Computes the sum along segments of a tensor.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_segment_sum."""
            data = input_nodes[0]
            segment_ids = input_nodes[1]
            
            # Custom segment_sum implementation
            def segment_sum_impl(data, segment_ids):
                # Get number of segments
                num_segments = segment_ids.max().item() + 1
                
                # Create output tensor
                output_shape = [num_segments] + list(data.shape[1:])
                output = torch.zeros(output_shape, dtype=data.dtype, device=data.device)
                
                # Sum data into segments
                for i in range(num_segments):
                    mask = segment_ids == i
                    output[i] = data[mask].sum(dim=0)
                
                return output
            
            output_node = graph.call_function(segment_sum_impl, args=(data, segment_ids))
            output_node.name = node_name
            return output_node
        return build_graph

    
    # Signal Processing Operations
    def _convert_rfft2d(self, inputs: List[Any], options: Dict[str, Any]) -> Callable:
        """
        Convert TFLite RFFT2D to PyTorch fft.rfft2.
        
        RFFT2D takes 2 inputs:
        - input tensor (signal)
        - fft_length tensor (shape [2])
        
        The fft_length needs to be converted to a tuple for PyTorch.
        """
        def build_graph(graph: Graph, input_nodes: List[Node], weights: Dict,
                       operator, subgraph, node_name: str, node_counter: Dict,
                       parameter_dict: Dict) -> Node:
            """Build FX graph for _convert_rfft2d."""
            signal = input_nodes[0]
            
            # If fft_length is provided, extract it; otherwise use None
            if len(input_nodes) > 1:
                fft_length = input_nodes[1]
                
                # Custom rfft2d implementation with fft_length
                def rfft2d_with_length(signal, fft_length):
                    if torch.is_tensor(fft_length):
                        s = tuple(fft_length.tolist())
                    else:
                        s = fft_length
                    return torch.fft.rfft2(signal, s=s)
                
                output_node = graph.call_function(rfft2d_with_length, args=(signal, fft_length))
            else:
                # No fft_length provided
                output_node = graph.call_function(torch.fft.rfft2, args=(signal,))
            
            output_node.name = node_name
            return output_node
        return build_graph


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
