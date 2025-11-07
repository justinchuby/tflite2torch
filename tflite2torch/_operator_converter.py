"""
Operator converter for mapping TFLite operators to PyTorch custom ops.

This module provides the OperatorConverter class that maps TFLite operator types
to their corresponding PyTorch custom operator implementations in tflite2torch.ops.
"""

from __future__ import annotations

import inspect
import torch


def _get_tfl_ops():
    """Get the tfl ops namespace, ensuring ops are registered."""
    # Import to ensure custom ops are registered
    import tflite2torch.ops  # noqa: F401
    return torch.ops.tfl


class OperatorConverter:
    """
    Converts TFLite operators to PyTorch FX graph nodes.

    This class maps TFLite operator types to their corresponding custom ops
    registered in the tfl namespace (torch.ops.tfl).
    """

    # Mapping from TFLite operator names to custom op function names
    OP_MAP = {
        "ADD": "add",
        "AVERAGE_POOL_2D": "average_pool_2d",
        "CONCATENATION": "concatenation",
        "CONV_2D": "conv_2d",
        "DEPTHWISE_CONV_2D": "depthwise_conv_2d",
        "DEPTH_TO_SPACE": "depth_to_space",
        "DEQUANTIZE": "dequantize",
        "EMBEDDING_LOOKUP": "embedding_lookup",
        "FLOOR": "floor",
        "FULLY_CONNECTED": "fully_connected",
        "HASHTABLE_LOOKUP": "hashtable_lookup",
        "L2_NORMALIZATION": "l2_normalization",
        "L2_POOL_2D": "l2_pool_2d",
        "LOCAL_RESPONSE_NORMALIZATION": "local_response_normalization",
        "LOGISTIC": "logistic",
        "LSH_PROJECTION": "lsh_projection",
        "LSTM": "lstm",
        "MAX_POOL_2D": "max_pool_2d",
        "MUL": "mul",
        "RELU": "relu",
        "RELU_N1_TO_1": "relu_n1_to_1",
        "RELU6": "relu6",
        "RESHAPE": "reshape",
        "RESIZE_BILINEAR": "resize_bilinear",
        "RNN": "rnn",
        "SOFTMAX": "softmax",
        "SPACE_TO_DEPTH": "space_to_depth",
        "SVDF": "svdf",
        "TANH": "tanh",
        "CONCAT_EMBEDDINGS": "concat_embeddings",
        "SKIP_GRAM": "skip_gram",
        "CALL": "call",
        "CUSTOM": "custom",
        "EMBEDDING_LOOKUP_SPARSE": "embedding_lookup_sparse",
        "PAD": "pad",
        "UNIDIRECTIONAL_SEQUENCE_RNN": "unidirectional_sequence_rnn",
        "GATHER": "gather",
        "BATCH_TO_SPACE_ND": "batch_to_space_nd",
        "SPACE_TO_BATCH_ND": "space_to_batch_nd",
        "TRANSPOSE": "transpose",
        "MEAN": "mean",
        "SUB": "sub",
        "DIV": "div",
        "SQUEEZE": "squeeze",
        "UNIDIRECTIONAL_SEQUENCE_LSTM": "unidirectional_sequence_lstm",
        "STRIDED_SLICE": "strided_slice",
        "BIDIRECTIONAL_SEQUENCE_RNN": "bidirectional_sequence_rnn",
        "EXP": "exp",
        "TOPK_V2": "topk_v2",
        "SPLIT": "split",
        "LOG_SOFTMAX": "log_softmax",
        "DELEGATE": "delegate",
        "BIDIRECTIONAL_SEQUENCE_LSTM": "bidirectional_sequence_lstm",
        "CAST": "cast",
        "PRELU": "prelu",
        "MAXIMUM": "maximum",
        "ARG_MAX": "arg_max",
        "MINIMUM": "minimum",
        "LESS": "less",
        "NEG": "neg",
        "PADV2": "padv2",
        "GREATER": "greater",
        "GREATER_EQUAL": "greater_equal",
        "LESS_EQUAL": "less_equal",
        "SELECT": "select",
        "SLICE": "slice",
        "SIN": "sin",
        "TRANSPOSE_CONV": "transpose_conv",
        "SPARSE_TO_DENSE": "sparse_to_dense",
        "TILE": "tile",
        "EXPAND_DIMS": "expand_dims",
        "EQUAL": "equal",
        "NOT_EQUAL": "not_equal",
        "LOG": "log",
        "SUM": "sum",
        "SQRT": "sqrt",
        "RSQRT": "rsqrt",
        "SHAPE": "shape",
        "POW": "pow",
        "ARG_MIN": "arg_min",
        "FAKE_QUANT": "fake_quant",
        "REDUCE_PROD": "reduce_prod",
        "REDUCE_MAX": "reduce_max",
        "PACK": "pack",
        "LOGICAL_OR": "logical_or",
        "ONE_HOT": "one_hot",
        "LOGICAL_AND": "logical_and",
        "LOGICAL_NOT": "logical_not",
        "UNPACK": "unpack",
        "REDUCE_MIN": "reduce_min",
        "FLOOR_DIV": "floor_div",
        "REDUCE_ANY": "reduce_any",
        "SQUARE": "square",
        "ZEROS_LIKE": "zeros_like",
        "FILL": "fill",
        "FLOOR_MOD": "floor_mod",
        "RANGE": "range",
        "RESIZE_NEAREST_NEIGHBOR": "resize_nearest_neighbor",
        "LEAKY_RELU": "leaky_relu",
        "SQUARED_DIFFERENCE": "squared_difference",
        "MIRROR_PAD": "mirror_pad",
        "ABS": "abs",
        "SPLIT_V": "split_v",
        "UNIQUE": "unique",
        "CEIL": "ceil",
        "REVERSE_V2": "reverse_v2",
        "ADD_N": "add_n",
        "GATHER_ND": "gather_nd",
        "COS": "cos",
        "WHERE": "where",
        "RANK": "rank",
        "ELU": "elu",
        "REVERSE_SEQUENCE": "reverse_sequence",
        "MATRIX_DIAG": "matrix_diag",
        "QUANTIZE": "quantize",
        "MATRIX_SET_DIAG": "matrix_set_diag",
        "ROUND": "round",
        "HARD_SWISH": "hard_swish",
        "IF": "if_op",
        "WHILE": "while_op",
        "NON_MAX_SUPPRESSION_V4": "non_max_suppression_v4",
        "NON_MAX_SUPPRESSION_V5": "non_max_suppression_v5",
        "SCATTER_ND": "scatter_nd",
        "SELECT_V2": "select_v2",
        "DENSIFY": "densify",
        "SEGMENT_SUM": "segment_sum",
        "BATCH_MATMUL": "batch_matmul",
        "PLACEHOLDER_FOR_GREATER_OP_CODES": None,  # Reserved
        "CUMSUM": "cumsum",
        "CALL_ONCE": "call_once",
        "BROADCAST_TO": "broadcast_to",
        "RFFT2D": "rfft2d",
        "CONV_3D": "conv_3d",
        "IMAG": "imag",
        "REAL": "real",
        "COMPLEX_ABS": "complex_abs",
        "HASHTABLE": "hashtable",
        "HASHTABLE_FIND": "hashtable_find",
        "HASHTABLE_IMPORT": "hashtable_import",
        "HASHTABLE_SIZE": "hashtable_size",
        "REDUCE_ALL": "reduce_all",
        "CONV_3D_TRANSPOSE": "conv_3d_transpose",
        "VAR_HANDLE": "var_handle",
        "READ_VARIABLE": "read_variable",
        "ASSIGN_VARIABLE": "assign_variable",
        "BROADCAST_ARGS": "broadcast_args",
        "RANDOM_STANDARD_NORMAL": "random_standard_normal",
        "BUCKETIZE": "bucketize",
        "RANDOM_UNIFORM": "random_uniform",
        "MULTINOMIAL": "multinomial",
        "GELU": "gelu",
        "DYNAMIC_UPDATE_SLICE": "dynamic_update_slice",
        "RELU_0_TO_1": "relu_0_to_1",
        "UNSORTED_SEGMENT_PROD": "unsorted_segment_prod",
        "UNSORTED_SEGMENT_MAX": "unsorted_segment_max",
        "UNSORTED_SEGMENT_SUM": "unsorted_segment_sum",
        "ATAN2": "atan2",
        "UNSORTED_SEGMENT_MIN": "unsorted_segment_min",
        "SIGN": "sign",
        "BITCAST": "bitcast",
        "BITWISE_XOR": "bitwise_xor",
        "RIGHT_SHIFT": "right_shift",
        "DILATE": "dilate",
    }

    def __init__(self):
        pass

    def convert(
        self,
        op_type: str,
        inputs: list[int],
        builtin_options: dict | None = None,
    ):
        """
        Convert a TFLite operator to a PyTorch FX graph builder function.

        Args:
            op_type: TFLite operator type (e.g., "CONV_2D", "ADD")
            inputs: List of input tensor indices
            builtin_options: Dictionary of operator options

        Returns:
            A callable that builds FX graph nodes for this operator

        Raises:
            NotImplementedError: If the operator type is not supported
        """
        # Get the tfl ops namespace
        tfl_ops = _get_tfl_ops()

        # Get the custom op function name
        op_func_name = self.OP_MAP.get(op_type)

        if op_func_name is None:
            raise NotImplementedError(f"Operator {op_type} not supported")

        # Get the custom op function
        if not hasattr(tfl_ops, op_func_name):
            raise NotImplementedError(
                f"Operator {op_type} (tfl::{op_func_name}) not implemented"
            )

        op_func = getattr(tfl_ops, op_func_name)

        # Return a graph builder function
        def build_graph_node(
            graph,
            input_nodes,
            weights,
            operator,
            subgraph,
            node_name,
            node_counter_dict,
            parameter_dict,
        ):
            """Build an FX graph node for this operator."""
            # Extract builtin_options to pass as kwargs
            kwargs = {}
            if builtin_options:
                # Pass all builtin options as kwargs
                # The custom op implementations will handle which ones they need
                kwargs = builtin_options.copy()

            # Call the custom op with input nodes and options
            # The custom ops are already registered with torch.library.custom_op
            # so we can call them directly through torch.ops.tfl
            output_node = graph.call_function(
                op_func,
                args=tuple(input_nodes),
                kwargs=kwargs,
            )
            output_node.name = node_name
            node_counter_dict["count"] += 1
            return output_node

        return build_graph_node
