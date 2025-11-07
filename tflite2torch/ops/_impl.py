"""Custom ops implementations for tflite ops in PyTorch.

This file implements custom ops for TFLite operators in the order they appear
in the TFLite BuiltinOperator enum (see plans/tflite_ops.md).
"""

import torch


# ============================================================================
# TFLite Operators (in order from plans/tflite_ops.md)
# ============================================================================

# 0: ADD
@torch.library.custom_op("tfl::add", mutates_args=())
def tfl_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.add(x, y)


@tfl_add.register_fake
def _(x, y):
    return torch.add(x, y)


# 1: AVERAGE_POOL_2D
@torch.library.custom_op("tfl::average_pool_2d", mutates_args=())
def tfl_average_pool_2d(x: torch.Tensor, kernel_size: list[int], stride: list[int], padding: str) -> torch.Tensor:
    # Simplified - full implementation would handle SAME/VALID padding
    return torch.nn.functional.avg_pool2d(x, kernel_size, stride)


@tfl_average_pool_2d.register_fake
def _(x, kernel_size, stride, padding):
    return torch.nn.functional.avg_pool2d(x, kernel_size, stride)


# 2: CONCATENATION
@torch.library.custom_op("tfl::concatenation", mutates_args=())
def tfl_concatenation(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


@tfl_concatenation.register_fake
def _(tensors, dim):
    return torch.cat(tensors, dim=dim)


# 3: CONV_2D
@torch.library.custom_op("tfl::conv_2d", mutates_args=())
def tfl_conv_2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: list[int], padding: str) -> torch.Tensor:
    # Simplified - full implementation would handle SAME/VALID padding
    return torch.nn.functional.conv2d(x, weight, bias, stride)


@tfl_conv_2d.register_fake
def _(x, weight, bias, stride, padding):
    return torch.nn.functional.conv2d(x, weight, bias, stride)


# 4: DEPTHWISE_CONV_2D
@torch.library.custom_op("tfl::depthwise_conv_2d", mutates_args=())
def tfl_depthwise_conv_2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: list[int], padding: str) -> torch.Tensor:
    # Simplified - full implementation would use groups parameter
    return torch.nn.functional.conv2d(x, weight, bias, stride)


@tfl_depthwise_conv_2d.register_fake
def _(x, weight, bias, stride, padding):
    return torch.nn.functional.conv2d(x, weight, bias, stride)


# 5: DEPTH_TO_SPACE
@torch.library.custom_op("tfl::depth_to_space", mutates_args=())
def tfl_depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.nn.functional.pixel_shuffle(x, block_size)


@tfl_depth_to_space.register_fake
def _(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


# 6: DEQUANTIZE
@torch.library.custom_op("tfl::dequantize", mutates_args=())
def tfl_dequantize(x: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    # Simplified implementation
    return x.float()


@tfl_dequantize.register_fake
def _(x, scale, zero_point):
    return x.float()


# 7: EMBEDDING_LOOKUP
@torch.library.custom_op("tfl::embedding_lookup", mutates_args=())
def tfl_embedding_lookup(params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.embedding(indices, params)


@tfl_embedding_lookup.register_fake
def _(params, indices):
    return torch.nn.functional.embedding(indices, params)


# 8: FLOOR
@torch.library.custom_op("tfl::floor", mutates_args=())
def tfl_floor(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x)


@tfl_floor.register_fake
def _(x):
    return torch.floor(x)

# 9: FULLY_CONNECTED
@torch.library.custom_op("tfl::fully_connected", mutates_args=())
def tfl_fully_connected(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


@tfl_fully_connected.register_fake
def _(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)


# 10: HASHTABLE_LOOKUP
@torch.library.custom_op("tfl::hashtable_lookup", mutates_args=())
def tfl_hashtable_lookup(keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    # Simplified implementation
    return values


@tfl_hashtable_lookup.register_fake
def _(keys, values, query):
    return values


# 11: L2_NORMALIZATION
@torch.library.custom_op("tfl::l2_normalization", mutates_args=())
def tfl_l2_normalization(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=dim)


@tfl_l2_normalization.register_fake
def _(x, dim):
    return torch.nn.functional.normalize(x, p=2, dim=dim)


# 12: L2_POOL_2D
@torch.library.custom_op("tfl::l2_pool_2d", mutates_args=())
def tfl_l2_pool_2d(x: torch.Tensor, kernel_size: list[int], stride: list[int]) -> torch.Tensor:
    # Simplified - L2 pooling: sqrt of avg of squares
    return torch.sqrt(torch.nn.functional.avg_pool2d(x * x, kernel_size, stride))


@tfl_l2_pool_2d.register_fake
def _(x, kernel_size, stride):
    return torch.sqrt(torch.nn.functional.avg_pool2d(x * x, kernel_size, stride))


# 13: LOCAL_RESPONSE_NORMALIZATION
@torch.library.custom_op("tfl::local_response_normalization", mutates_args=())
def tfl_local_response_normalization(x: torch.Tensor, radius: int, bias: float, alpha: float, beta: float) -> torch.Tensor:
    return torch.nn.functional.local_response_norm(x, 2 * radius + 1, alpha, beta, bias)


@tfl_local_response_normalization.register_fake
def _(x, radius, bias, alpha, beta):
    return torch.nn.functional.local_response_norm(x, 2 * radius + 1, alpha, beta, bias)


# 14: LOGISTIC
@torch.library.custom_op("tfl::logistic", mutates_args=())
def tfl_logistic(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@tfl_logistic.register_fake
def _(x):
    return torch.sigmoid(x)


# 15: LSH_PROJECTION
@torch.library.custom_op("tfl::lsh_projection", mutates_args=())
def tfl_lsh_projection(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # Simplified - Locality Sensitive Hashing projection
    return x


@tfl_lsh_projection.register_fake
def _(x, weights):
    return x


# 16: LSTM
@torch.library.custom_op("tfl::lstm", mutates_args=())
def tfl_lstm(x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Simplified - would need full LSTM implementation
    return x, hidden, cell


@tfl_lstm.register_fake
def _(x, hidden, cell, weights):
    return x, hidden, cell


# 17: MAX_POOL_2D
@torch.library.custom_op("tfl::max_pool_2d", mutates_args=())
def tfl_max_pool_2d(x: torch.Tensor, kernel_size: list[int], stride: list[int], padding: str) -> torch.Tensor:
    # Simplified - full implementation would handle SAME/VALID padding
    return torch.nn.functional.max_pool2d(x, kernel_size, stride)


@tfl_max_pool_2d.register_fake
def _(x, kernel_size, stride, padding):
    return torch.nn.functional.max_pool2d(x, kernel_size, stride)


# 18: MUL
@torch.library.custom_op("tfl::mul", mutates_args=())
def tfl_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mul(x, y)


@tfl_mul.register_fake
def _(x, y):
    return torch.mul(x, y)


# 19: RELU
@torch.library.custom_op("tfl::relu", mutates_args=())
def tfl_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)


@tfl_relu.register_fake
def _(x):
    return torch.nn.functional.relu(x)


# 20: RELU_N1_TO_1
@torch.library.custom_op("tfl::relu_n1_to_1", mutates_args=())
def tfl_relu_n1_to_1(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=-1.0, max=1.0)


@tfl_relu_n1_to_1.register_fake
def _(x):
    return torch.clamp(x, min=-1.0, max=1.0)


# 21: RELU6
@torch.library.custom_op("tfl::relu6", mutates_args=())
def tfl_relu6(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu6(x)


@tfl_relu6.register_fake
def _(x):
    return torch.nn.functional.relu6(x)


# 22: RESHAPE
@torch.library.custom_op("tfl::reshape", mutates_args=())
def tfl_reshape(x: torch.Tensor, shape: list[int]) -> torch.Tensor:
    return torch.reshape(x, shape)


@tfl_reshape.register_fake
def _(x, shape):
    return torch.reshape(x, shape)


# 23: RESIZE_BILINEAR
@torch.library.custom_op("tfl::resize_bilinear", mutates_args=())
def tfl_resize_bilinear(x: torch.Tensor, size: list[int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)


@tfl_resize_bilinear.register_fake
def _(x, size):
    return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)


# 24: RNN
@torch.library.custom_op("tfl::rnn", mutates_args=())
def tfl_rnn(x: torch.Tensor, hidden: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Simplified - would need full RNN implementation
    return x, hidden


@tfl_rnn.register_fake
def _(x, hidden, weights):
    return x, hidden


# 25: SOFTMAX
@torch.library.custom_op("tfl::softmax", mutates_args=())
def tfl_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=dim)


@tfl_softmax.register_fake
def _(x, dim):
    return torch.nn.functional.softmax(x, dim=dim)


# 26: SPACE_TO_DEPTH
@torch.library.custom_op("tfl::space_to_depth", mutates_args=())
def tfl_space_to_depth(x: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.nn.functional.pixel_unshuffle(x, block_size)


@tfl_space_to_depth.register_fake
def _(x, block_size):
    return torch.nn.functional.pixel_unshuffle(x, block_size)


# 27: SVDF
@torch.library.custom_op("tfl::svdf", mutates_args=())
def tfl_svdf(x: torch.Tensor, weights_feature: torch.Tensor, weights_time: torch.Tensor) -> torch.Tensor:
    # Simplified - Singular Value Decomposition Filter
    return x


@tfl_svdf.register_fake
def _(x, weights_feature, weights_time):
    return x


# 28: TANH
@torch.library.custom_op("tfl::tanh", mutates_args=())
def tfl_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


@tfl_tanh.register_fake
def _(x):
    return torch.tanh(x)


# 29: CONCAT_EMBEDDINGS
@torch.library.custom_op("tfl::concat_embeddings", mutates_args=())
def tfl_concat_embeddings(embeddings: list[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.cat(embeddings, dim=dim)


@tfl_concat_embeddings.register_fake
def _(embeddings, dim):
    return torch.cat(embeddings, dim=dim)


# 30: SKIP_GRAM
@torch.library.custom_op("tfl::skip_gram", mutates_args=())
def tfl_skip_gram(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # Simplified - Skip-gram model operation
    return x


@tfl_skip_gram.register_fake
def _(x, weights):
    return x


# 31: CALL
@torch.library.custom_op("tfl::call", mutates_args=())
def tfl_call(x: torch.Tensor) -> torch.Tensor:
    # Simplified - Call subgraph operation
    return x


@tfl_call.register_fake
def _(x):
    return x


# 32: CUSTOM
@torch.library.custom_op("tfl::custom", mutates_args=())
def tfl_custom(x: torch.Tensor) -> torch.Tensor:
    # Placeholder for custom operations
    return x


@tfl_custom.register_fake
def _(x):
    return x


# 33: EMBEDDING_LOOKUP_SPARSE
@torch.library.custom_op("tfl::embedding_lookup_sparse", mutates_args=())
def tfl_embedding_lookup_sparse(params: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # Simplified implementation
    return torch.nn.functional.embedding(indices, params)


@tfl_embedding_lookup_sparse.register_fake
def _(params, indices, weights):
    return torch.nn.functional.embedding(indices, params)


# 34: PAD
@torch.library.custom_op("tfl::pad", mutates_args=())
def tfl_pad(x: torch.Tensor, paddings: torch.Tensor) -> torch.Tensor:
    # Convert paddings tensor to padding list
    pad_list = paddings.flatten().tolist()
    return torch.nn.functional.pad(x, pad_list)


@tfl_pad.register_fake
def _(x, paddings):
    pad_list = paddings.flatten().tolist()
    return torch.nn.functional.pad(x, pad_list)


# 35: UNIDIRECTIONAL_SEQUENCE_RNN
@torch.library.custom_op("tfl::unidirectional_sequence_rnn", mutates_args=())
def tfl_unidirectional_sequence_rnn(x: torch.Tensor, hidden: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Simplified - would need full RNN implementation
    return x, hidden


@tfl_unidirectional_sequence_rnn.register_fake
def _(x, hidden, weights):
    return x, hidden


# 36: GATHER
@torch.library.custom_op("tfl::gather", mutates_args=())
def tfl_gather(params: torch.Tensor, indices: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.gather(params, axis, indices)


@tfl_gather.register_fake
def _(params, indices, axis):
    return torch.gather(params, axis, indices)


# 37: BATCH_TO_SPACE_ND
@torch.library.custom_op("tfl::batch_to_space_nd", mutates_args=())
def tfl_batch_to_space_nd(x: torch.Tensor, block_shape: list[int], crops: list[list[int]]) -> torch.Tensor:
    # Simplified implementation - full version would need more complex logic
    return x


@tfl_batch_to_space_nd.register_fake
def _(x, block_shape, crops):
    return x


# 38: SPACE_TO_BATCH_ND
@torch.library.custom_op("tfl::space_to_batch_nd", mutates_args=())
def tfl_space_to_batch_nd(x: torch.Tensor, block_shape: list[int], paddings: list[list[int]]) -> torch.Tensor:
    # Simplified implementation
    return x


@tfl_space_to_batch_nd.register_fake
def _(x, block_shape, paddings):
    return x


# 39: TRANSPOSE
@torch.library.custom_op("tfl::transpose", mutates_args=())
def tfl_transpose(x: torch.Tensor, perm: list[int]) -> torch.Tensor:
    return torch.permute(x, perm)


@tfl_transpose.register_fake
def _(x, perm):
    return torch.permute(x, perm)


# 40: MEAN
@torch.library.custom_op("tfl::mean", mutates_args=())
def tfl_mean(x: torch.Tensor, dim: list[int], keepdim: bool) -> torch.Tensor:
    return torch.mean(x, dim=dim, keepdim=keepdim)


@tfl_mean.register_fake
def _(x, dim, keepdim):
    return torch.mean(x, dim=dim, keepdim=keepdim)


# 41: SUB
@torch.library.custom_op("tfl::sub", mutates_args=())
def tfl_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sub(x, y)


@tfl_sub.register_fake
def _(x, y):
    return torch.sub(x, y)


# 42: DIV
@torch.library.custom_op("tfl::div", mutates_args=())
def tfl_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.div(x, y)


@tfl_div.register_fake
def _(x, y):
    return torch.div(x, y)


# 43: SQUEEZE
@torch.library.custom_op("tfl::squeeze", mutates_args=())
def tfl_squeeze(x: torch.Tensor, dims: list[int]) -> torch.Tensor:
    result = x
    for dim in sorted(dims, reverse=True):
        result = torch.squeeze(result, dim)
    return result


@tfl_squeeze.register_fake
def _(x, dims):
    result = x
    for dim in sorted(dims, reverse=True):
        result = torch.squeeze(result, dim)
    return result


# 44: UNIDIRECTIONAL_SEQUENCE_LSTM
@torch.library.custom_op("tfl::unidirectional_sequence_lstm", mutates_args=())
def tfl_unidirectional_sequence_lstm(x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Simplified - would need full LSTM implementation
    return x, hidden, cell


@tfl_unidirectional_sequence_lstm.register_fake
def _(x, hidden, cell, weights):
    return x, hidden, cell


# 45: STRIDED_SLICE
@torch.library.custom_op("tfl::strided_slice", mutates_args=())
def tfl_strided_slice(x: torch.Tensor, begin: list[int], end: list[int], strides: list[int]) -> torch.Tensor:
    # Simplified implementation
    return x


@tfl_strided_slice.register_fake
def _(x, begin, end, strides):
    return x


# 46: BIDIRECTIONAL_SEQUENCE_RNN
@torch.library.custom_op("tfl::bidirectional_sequence_rnn", mutates_args=())
def tfl_bidirectional_sequence_rnn(x: torch.Tensor, fw_hidden: torch.Tensor, bw_hidden: torch.Tensor, fw_weights: torch.Tensor, bw_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Simplified - would need full bidirectional RNN implementation
    return x, fw_hidden, bw_hidden


@tfl_bidirectional_sequence_rnn.register_fake
def _(x, fw_hidden, bw_hidden, fw_weights, bw_weights):
    return x, fw_hidden, bw_hidden


# 47: EXP
@torch.library.custom_op("tfl::exp", mutates_args=())
def tfl_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


@tfl_exp.register_fake
def _(x):
    return torch.exp(x)


# 48: TOPK_V2
@torch.library.custom_op("tfl::topk_v2", mutates_args=())
def tfl_topk_v2(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(x, k)


@tfl_topk_v2.register_fake
def _(x, k):
    return torch.topk(x, k)


# 49: SPLIT
@torch.library.custom_op("tfl::split", mutates_args=())
def tfl_split(x: torch.Tensor, num_splits: int, dim: int) -> list[torch.Tensor]:
    return list(torch.split(x, x.shape[dim] // num_splits, dim=dim))


@tfl_split.register_fake
def _(x, num_splits, dim):
    return list(torch.split(x, x.shape[dim] // num_splits, dim=dim))


# 50: LOG_SOFTMAX
@torch.library.custom_op("tfl::log_softmax", mutates_args=())
def tfl_log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.nn.functional.log_softmax(x, dim=dim)


@tfl_log_softmax.register_fake
def _(x, dim):
    return torch.nn.functional.log_softmax(x, dim=dim)


# 51: DELEGATE
@torch.library.custom_op("tfl::delegate", mutates_args=())
def tfl_delegate(x: torch.Tensor) -> torch.Tensor:
    # Placeholder for delegate operations
    return x


@tfl_delegate.register_fake
def _(x):
    return x


# 52: BIDIRECTIONAL_SEQUENCE_LSTM
@torch.library.custom_op("tfl::bidirectional_sequence_lstm", mutates_args=())
def tfl_bidirectional_sequence_lstm(x: torch.Tensor, fw_hidden: torch.Tensor, fw_cell: torch.Tensor, bw_hidden: torch.Tensor, bw_cell: torch.Tensor, fw_weights: torch.Tensor, bw_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Simplified - would need full bidirectional LSTM implementation
    return x, fw_hidden, fw_cell, bw_hidden, bw_cell


@tfl_bidirectional_sequence_lstm.register_fake
def _(x, fw_hidden, fw_cell, bw_hidden, bw_cell, fw_weights, bw_weights):
    return x, fw_hidden, fw_cell, bw_hidden, bw_cell


# 53: CAST
@torch.library.custom_op("tfl::cast", mutates_args=())
def tfl_cast(x: torch.Tensor, dtype: int) -> torch.Tensor:
    # dtype mapping would be needed here
    return x


@tfl_cast.register_fake
def _(x, dtype):
    return x


# 54: PRELU
@torch.library.custom_op("tfl::prelu", mutates_args=())
def tfl_prelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.prelu(x, weight)


@tfl_prelu.register_fake
def _(x, weight):
    return torch.nn.functional.prelu(x, weight)


# 55: MAXIMUM
@torch.library.custom_op("tfl::maximum", mutates_args=())
def tfl_maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, y)


@tfl_maximum.register_fake
def _(x, y):
    return torch.maximum(x, y)


# 56: ARG_MAX
@torch.library.custom_op("tfl::arg_max", mutates_args=())
def tfl_arg_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.argmax(x, dim=dim)


@tfl_arg_max.register_fake
def _(x, dim):
    return torch.argmax(x, dim=dim)


# 57: MINIMUM
@torch.library.custom_op("tfl::minimum", mutates_args=())
def tfl_minimum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.minimum(x, y)


@tfl_minimum.register_fake
def _(x, y):
    return torch.minimum(x, y)


# 58: LESS
@torch.library.custom_op("tfl::less", mutates_args=())
def tfl_less(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.lt(x, y)


@tfl_less.register_fake
def _(x, y):
    return torch.lt(x, y)


# 59: NEG
@torch.library.custom_op("tfl::neg", mutates_args=())
def tfl_neg(x: torch.Tensor) -> torch.Tensor:
    return torch.neg(x)


@tfl_neg.register_fake
def _(x):
    return torch.neg(x)


# 60: PADV2
@torch.library.custom_op("tfl::padv2", mutates_args=())
def tfl_padv2(x: torch.Tensor, paddings: torch.Tensor, constant_values: float) -> torch.Tensor:
    pad_list = paddings.flatten().tolist()
    return torch.nn.functional.pad(x, pad_list, value=constant_values)


@tfl_padv2.register_fake
def _(x, paddings, constant_values):
    pad_list = paddings.flatten().tolist()
    return torch.nn.functional.pad(x, pad_list, value=constant_values)


# 61: GREATER
@torch.library.custom_op("tfl::greater", mutates_args=())
def tfl_greater(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.gt(x, y)


@tfl_greater.register_fake
def _(x, y):
    return torch.gt(x, y)


# 62: GREATER_EQUAL
@torch.library.custom_op("tfl::greater_equal", mutates_args=())
def tfl_greater_equal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.ge(x, y)


@tfl_greater_equal.register_fake
def _(x, y):
    return torch.ge(x, y)


# 63: LESS_EQUAL
@torch.library.custom_op("tfl::less_equal", mutates_args=())
def tfl_less_equal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.le(x, y)


@tfl_less_equal.register_fake
def _(x, y):
    return torch.le(x, y)


# 64: SELECT
@torch.library.custom_op("tfl::select", mutates_args=())
def tfl_select(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(condition, x, y)


@tfl_select.register_fake
def _(condition, x, y):
    return torch.where(condition, x, y)


# 65: SLICE
@torch.library.custom_op("tfl::slice", mutates_args=())
def tfl_slice(x: torch.Tensor, begin: list[int], size: list[int]) -> torch.Tensor:
    # Simplified implementation
    return x


@tfl_slice.register_fake
def _(x, begin, size):
    return x


# 66: SIN
@torch.library.custom_op("tfl::sin", mutates_args=())
def tfl_sin(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


@tfl_sin.register_fake
def _(x):
    return torch.sin(x)


# 67: TRANSPOSE_CONV
@torch.library.custom_op("tfl::transpose_conv", mutates_args=())
def tfl_transpose_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, output_shape: list[int], stride: list[int]) -> torch.Tensor:
    # Simplified - would need full transpose convolution implementation
    return torch.nn.functional.conv_transpose2d(x, weight, bias, stride)


@tfl_transpose_conv.register_fake
def _(x, weight, bias, output_shape, stride):
    return torch.nn.functional.conv_transpose2d(x, weight, bias, stride)


# 68: SPARSE_TO_DENSE
@torch.library.custom_op("tfl::sparse_to_dense", mutates_args=())
def tfl_sparse_to_dense(sparse_indices: torch.Tensor, output_shape: list[int], sparse_values: torch.Tensor, default_value: float) -> torch.Tensor:
    return torch.full(output_shape, default_value)


@tfl_sparse_to_dense.register_fake
def _(sparse_indices, output_shape, sparse_values, default_value):
    return torch.full(output_shape, default_value)


# 69: TILE
@torch.library.custom_op("tfl::tile", mutates_args=())
def tfl_tile(x: torch.Tensor, multiples: list[int]) -> torch.Tensor:
    return x.repeat(multiples)


@tfl_tile.register_fake
def _(x, multiples):
    return x.repeat(multiples)


# 70: EXPAND_DIMS
@torch.library.custom_op("tfl::expand_dims", mutates_args=())
def tfl_expand_dims(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.unsqueeze(x, dim)


@tfl_expand_dims.register_fake
def _(x, dim):
    return torch.unsqueeze(x, dim)


# 71: EQUAL
@torch.library.custom_op("tfl::equal", mutates_args=())
def tfl_equal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.eq(x, y)


@tfl_equal.register_fake
def _(x, y):
    return torch.eq(x, y)


# 72: NOT_EQUAL
@torch.library.custom_op("tfl::not_equal", mutates_args=())
def tfl_not_equal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.ne(x, y)


@tfl_not_equal.register_fake
def _(x, y):
    return torch.ne(x, y)


# 73: LOG
@torch.library.custom_op("tfl::log", mutates_args=())
def tfl_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


@tfl_log.register_fake
def _(x):
    return torch.log(x)


# 74: SUM
@torch.library.custom_op("tfl::sum", mutates_args=())
def tfl_sum(x: torch.Tensor, dim: list[int], keepdim: bool) -> torch.Tensor:
    return torch.sum(x, dim=dim, keepdim=keepdim)


@tfl_sum.register_fake
def _(x, dim, keepdim):
    return torch.sum(x, dim=dim, keepdim=keepdim)


# 75: SQRT
@torch.library.custom_op("tfl::sqrt", mutates_args=())
def tfl_sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x)


@tfl_sqrt.register_fake
def _(x):
    return torch.sqrt(x)


# 76: RSQRT
@torch.library.custom_op("tfl::rsqrt", mutates_args=())
def tfl_rsqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.rsqrt(x)


@tfl_rsqrt.register_fake
def _(x):
    return torch.rsqrt(x)


# 77: SHAPE
@torch.library.custom_op("tfl::shape", mutates_args=())
def tfl_shape(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(list(x.shape))


@tfl_shape.register_fake
def _(x):
    return torch.tensor(list(x.shape))


# 78: POW
@torch.library.custom_op("tfl::pow", mutates_args=())
def tfl_pow(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.pow(x, y)


@tfl_pow.register_fake
def _(x, y):
    return torch.pow(x, y)


# 79: ARG_MIN
@torch.library.custom_op("tfl::arg_min", mutates_args=())
def tfl_arg_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.argmin(x, dim=dim)


@tfl_arg_min.register_fake
def _(x, dim):
    return torch.argmin(x, dim=dim)


# 80: FAKE_QUANT
@torch.library.custom_op("tfl::fake_quant", mutates_args=())
def tfl_fake_quant(x: torch.Tensor, min_val: float, max_val: float, num_bits: int) -> torch.Tensor:
    # Simplified fake quantization
    return torch.fake_quantize_per_tensor_affine(x, (max_val - min_val) / (2 ** num_bits - 1), 0, 0, 2 ** num_bits - 1)


@tfl_fake_quant.register_fake
def _(x, min_val, max_val, num_bits):
    return torch.fake_quantize_per_tensor_affine(x, (max_val - min_val) / (2 ** num_bits - 1), 0, 0, 2 ** num_bits - 1)



# 81: REDUCE_PROD
@torch.library.custom_op("tfl::reduce_prod", mutates_args=())
def tfl_reduce_prod(x: torch.Tensor, dim: list[int], keepdim: bool) -> torch.Tensor:
    result = x
    for d in sorted(dim, reverse=True):
        result = torch.prod(result, dim=d, keepdim=keepdim)
    return result


@tfl_reduce_prod.register_fake
def _(x, dim, keepdim):
    result = x
    for d in sorted(dim, reverse=True):
        result = torch.prod(result, dim=d, keepdim=keepdim)
    return result


# 82: REDUCE_MAX
@torch.library.custom_op("tfl::reduce_max", mutates_args=())
def tfl_reduce_max(x: torch.Tensor, dim: list[int], keepdim: bool) -> torch.Tensor:
    result = x
    for d in sorted(dim, reverse=True):
        result = torch.amax(result, dim=d, keepdim=keepdim)
    return result


@tfl_reduce_max.register_fake
def _(x, dim, keepdim):
    result = x
    for d in sorted(dim, reverse=True):
        result = torch.amax(result, dim=d, keepdim=keepdim)
    return result


# 83: PACK
@torch.library.custom_op("tfl::pack", mutates_args=())
def tfl_pack(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.stack(tensors, dim=dim)


@tfl_pack.register_fake
def _(tensors, dim):
    return torch.stack(tensors, dim=dim)


# 84: LOGICAL_OR
@torch.library.custom_op("tfl::logical_or", mutates_args=())
def tfl_logical_or(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.logical_or(x, y)


@tfl_logical_or.register_fake
def _(x, y):
    return torch.logical_or(x, y)


# 85: ONE_HOT
@torch.library.custom_op("tfl::one_hot", mutates_args=())
def tfl_one_hot(indices: torch.Tensor, depth: int, on_value: float, off_value: float) -> torch.Tensor:
    result = torch.nn.functional.one_hot(indices, num_classes=depth)
    return result * on_value + (1 - result) * off_value


@tfl_one_hot.register_fake
def _(indices, depth, on_value, off_value):
    result = torch.nn.functional.one_hot(indices, num_classes=depth)
    return result * on_value + (1 - result) * off_value


# 86: LOGICAL_AND
@torch.library.custom_op("tfl::logical_and", mutates_args=())
def tfl_logical_and(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.logical_and(x, y)


@tfl_logical_and.register_fake
def _(x, y):
    return torch.logical_and(x, y)


# 87: LOGICAL_NOT
@torch.library.custom_op("tfl::logical_not", mutates_args=())
def tfl_logical_not(x: torch.Tensor) -> torch.Tensor:
    return torch.logical_not(x)


@tfl_logical_not.register_fake
def _(x):
    return torch.logical_not(x)


# 88: UNPACK
@torch.library.custom_op("tfl::unpack", mutates_args=())
def tfl_unpack(x: torch.Tensor, num: int, axis: int) -> list[torch.Tensor]:
    return list(torch.unbind(x, dim=axis))


@tfl_unpack.register_fake
def _(x, num, axis):
    return list(torch.unbind(x, dim=axis))


# 89: REDUCE_MIN
@torch.library.custom_op("tfl::reduce_min", mutates_args=())
def tfl_reduce_min(x: torch.Tensor, dim: list[int], keepdim: bool) -> torch.Tensor:
    result = x
    for d in sorted(dim, reverse=True):
        result = torch.amin(result, dim=d, keepdim=keepdim)
    return result


@tfl_reduce_min.register_fake
def _(x, dim, keepdim):
    result = x
    for d in sorted(dim, reverse=True):
        result = torch.amin(result, dim=d, keepdim=keepdim)
    return result


# 90: FLOOR_DIV
@torch.library.custom_op("tfl::floor_div", mutates_args=())
def tfl_floor_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.floor_divide(x, y)


@tfl_floor_div.register_fake
def _(x, y):
    return torch.floor_divide(x, y)


# 91: REDUCE_ANY
@torch.library.custom_op("tfl::reduce_any", mutates_args=())
def tfl_reduce_any(x: torch.Tensor, dim: list[int], keepdim: bool) -> torch.Tensor:
    return torch.any(x, dim=dim[0] if len(dim) == 1 else None, keepdim=keepdim)


@tfl_reduce_any.register_fake
def _(x, dim, keepdim):
    return torch.any(x, dim=dim[0] if len(dim) == 1 else None, keepdim=keepdim)


# 92: SQUARE
@torch.library.custom_op("tfl::square", mutates_args=())
def tfl_square(x: torch.Tensor) -> torch.Tensor:
    return torch.square(x)


@tfl_square.register_fake
def _(x):
    return torch.square(x)


# 93: ZEROS_LIKE
@torch.library.custom_op("tfl::zeros_like", mutates_args=())
def tfl_zeros_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


@tfl_zeros_like.register_fake
def _(x):
    return torch.zeros_like(x)


# 94: FILL
@torch.library.custom_op("tfl::fill", mutates_args=())
def tfl_fill(shape: list[int], value: float) -> torch.Tensor:
    return torch.full(shape, value)


@tfl_fill.register_fake
def _(shape, value):
    return torch.full(shape, value)


# 95: FLOOR_MOD
@torch.library.custom_op("tfl::floor_mod", mutates_args=())
def tfl_floor_mod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.fmod(torch.floor(x), y)


@tfl_floor_mod.register_fake
def _(x, y):
    return torch.fmod(torch.floor(x), y)


# 96: RANGE
@torch.library.custom_op("tfl::range", mutates_args=())
def tfl_range(start: float, limit: float, delta: float) -> torch.Tensor:
    return torch.arange(start, limit, delta)


@tfl_range.register_fake
def _(start, limit, delta):
    return torch.arange(start, limit, delta)


# 97: RESIZE_NEAREST_NEIGHBOR
@torch.library.custom_op("tfl::resize_nearest_neighbor", mutates_args=())
def tfl_resize_nearest_neighbor(x: torch.Tensor, size: list[int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(x, size=size, mode='nearest')


@tfl_resize_nearest_neighbor.register_fake
def _(x, size):
    return torch.nn.functional.interpolate(x, size=size, mode='nearest')


# 98: LEAKY_RELU
@torch.library.custom_op("tfl::leaky_relu", mutates_args=())
def tfl_leaky_relu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=alpha)


@tfl_leaky_relu.register_fake
def _(x, alpha):
    return torch.nn.functional.leaky_relu(x, negative_slope=alpha)


# 99: SQUARED_DIFFERENCE
@torch.library.custom_op("tfl::squared_difference", mutates_args=())
def tfl_squared_difference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff = torch.sub(x, y)
    return torch.square(diff)


@tfl_squared_difference.register_fake
def _(x, y):
    diff = torch.sub(x, y)
    return torch.square(diff)


# 100: MIRROR_PAD
@torch.library.custom_op("tfl::mirror_pad", mutates_args=())
def tfl_mirror_pad(x: torch.Tensor, paddings: torch.Tensor, mode: str) -> torch.Tensor:
    # Simplified implementation
    return x


@tfl_mirror_pad.register_fake
def _(x, paddings, mode):
    return x


# 101: ABS
@torch.library.custom_op("tfl::abs", mutates_args=())
def tfl_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


@tfl_abs.register_fake
def _(x):
    return torch.abs(x)


# 102: SPLIT_V
@torch.library.custom_op("tfl::split_v", mutates_args=())
def tfl_split_v(x: torch.Tensor, size_splits: list[int], dim: int) -> list[torch.Tensor]:
    return list(torch.split(x, size_splits, dim=dim))


@tfl_split_v.register_fake
def _(x, size_splits, dim):
    return list(torch.split(x, size_splits, dim=dim))


# 103: UNIQUE
@torch.library.custom_op("tfl::unique", mutates_args=())
def tfl_unique(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.unique(x, return_inverse=True)


@tfl_unique.register_fake
def _(x):
    return torch.unique(x, return_inverse=True)


# 104: CEIL
@torch.library.custom_op("tfl::ceil", mutates_args=())
def tfl_ceil(x: torch.Tensor) -> torch.Tensor:
    return torch.ceil(x)


@tfl_ceil.register_fake
def _(x):
    return torch.ceil(x)


# 105: REVERSE_V2
@torch.library.custom_op("tfl::reverse_v2", mutates_args=())
def tfl_reverse_v2(x: torch.Tensor, axis: list[int]) -> torch.Tensor:
    result = x
    for ax in axis:
        result = torch.flip(result, dims=[ax])
    return result


@tfl_reverse_v2.register_fake
def _(x, axis):
    result = x
    for ax in axis:
        result = torch.flip(result, dims=[ax])
    return result


# 106: ADD_N
@torch.library.custom_op("tfl::add_n", mutates_args=())
def tfl_add_n(inputs: list[torch.Tensor]) -> torch.Tensor:
    result = inputs[0]
    for tensor in inputs[1:]:
        result = torch.add(result, tensor)
    return result


@tfl_add_n.register_fake
def _(inputs):
    result = inputs[0]
    for tensor in inputs[1:]:
        result = torch.add(result, tensor)
    return result


# 107: GATHER_ND
@torch.library.custom_op("tfl::gather_nd", mutates_args=())
def tfl_gather_nd(params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # Simplified implementation
    return params


@tfl_gather_nd.register_fake
def _(params, indices):
    return params


# 108: COS
@torch.library.custom_op("tfl::cos", mutates_args=())
def tfl_cos(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x)


@tfl_cos.register_fake
def _(x):
    return torch.cos(x)


# 109: WHERE
@torch.library.custom_op("tfl::where", mutates_args=())
def tfl_where(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(condition, x, y)


@tfl_where.register_fake
def _(condition, x, y):
    return torch.where(condition, x, y)


# 110: RANK
@torch.library.custom_op("tfl::rank", mutates_args=())
def tfl_rank(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(x.dim())


@tfl_rank.register_fake
def _(x):
    return torch.tensor(x.dim())


# 111: ELU
@torch.library.custom_op("tfl::elu", mutates_args=())
def tfl_elu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.elu(x)


@tfl_elu.register_fake
def _(x):
    return torch.nn.functional.elu(x)


# 112: REVERSE_SEQUENCE
@torch.library.custom_op("tfl::reverse_sequence", mutates_args=())
def tfl_reverse_sequence(x: torch.Tensor, seq_lengths: torch.Tensor, seq_dim: int, batch_dim: int) -> torch.Tensor:
    # Simplified implementation
    return x


@tfl_reverse_sequence.register_fake
def _(x, seq_lengths, seq_dim, batch_dim):
    return x


# 113: MATRIX_DIAG
@torch.library.custom_op("tfl::matrix_diag", mutates_args=())
def tfl_matrix_diag(diagonal: torch.Tensor) -> torch.Tensor:
    return torch.diag_embed(diagonal)


@tfl_matrix_diag.register_fake
def _(diagonal):
    return torch.diag_embed(diagonal)


# 114: QUANTIZE
# (Placeholder for quantization operation)


# 115: MATRIX_SET_DIAG
@torch.library.custom_op("tfl::matrix_set_diag", mutates_args=())
def tfl_matrix_set_diag(x: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    # Simplified implementation
    return x


@tfl_matrix_set_diag.register_fake
def _(x, diagonal):
    return x


# 116: ROUND
@torch.library.custom_op("tfl::round", mutates_args=())
def tfl_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x)


@tfl_round.register_fake
def _(x):
    return torch.round(x)


# 117: HARD_SWISH
@torch.library.custom_op("tfl::hard_swish", mutates_args=())
def tfl_hard_swish(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.hardswish(x)


@tfl_hard_swish.register_fake
def _(x):
    return torch.nn.functional.hardswish(x)


# 118: IF
# (Placeholder for control flow operation)


# 119: WHILE
# (Placeholder for control flow operation)


# 120: NON_MAX_SUPPRESSION_V4
# (Placeholder for NMS operation)


# 121: NON_MAX_SUPPRESSION_V5
# (Placeholder for NMS operation)


# 122: SCATTER_ND
@torch.library.custom_op("tfl::scatter_nd", mutates_args=())
def tfl_scatter_nd(indices: torch.Tensor, updates: torch.Tensor, shape: list[int]) -> torch.Tensor:
    # Simplified implementation
    return torch.zeros(shape)


@tfl_scatter_nd.register_fake
def _(indices, updates, shape):
    return torch.zeros(shape)


# 123: SELECT_V2
@torch.library.custom_op("tfl::select_v2", mutates_args=())
def tfl_select_v2(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(condition, x, y)


@tfl_select_v2.register_fake
def _(condition, x, y):
    return torch.where(condition, x, y)


# 124: DENSIFY
# (Placeholder for sparse tensor operation)


# 125: SEGMENT_SUM
@torch.library.custom_op("tfl::segment_sum", mutates_args=())
def tfl_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    # Simplified implementation
    return data


@tfl_segment_sum.register_fake
def _(data, segment_ids):
    return data


# 126: BATCH_MATMUL
# (Placeholder - could use torch.bmm)


# 127: PLACEHOLDER_FOR_GREATER_OP_CODES
# (Reserved placeholder)


# 128: CUMSUM
@torch.library.custom_op("tfl::cumsum", mutates_args=())
def tfl_cumsum(x: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.cumsum(x, dim=axis)


@tfl_cumsum.register_fake
def _(x, axis):
    return torch.cumsum(x, dim=axis)


# 129: CALL_ONCE
# (Placeholder for control flow operation)


# 130: BROADCAST_TO
@torch.library.custom_op("tfl::broadcast_to", mutates_args=())
def tfl_broadcast_to(x: torch.Tensor, shape: list[int]) -> torch.Tensor:
    return torch.broadcast_to(x, shape)


@tfl_broadcast_to.register_fake
def _(x, shape):
    return torch.broadcast_to(x, shape)


# 131: RFFT2D
# (Placeholder for FFT operation)


# 132: CONV_3D
# (Placeholder for 3D convolution)


# 133: IMAG
# (Placeholder for complex number operation)


# 134: REAL
# (Placeholder for complex number operation)


# 135: COMPLEX_ABS
# (Placeholder for complex number operation)


# 136: HASHTABLE
# (Placeholder for hashtable operation)


# 137: HASHTABLE_FIND
# (Placeholder for hashtable operation)


# 138: HASHTABLE_IMPORT
# (Placeholder for hashtable operation)


# 139: HASHTABLE_SIZE
# (Placeholder for hashtable operation)


# 140: REDUCE_ALL
@torch.library.custom_op("tfl::reduce_all", mutates_args=())
def tfl_reduce_all(x: torch.Tensor, dim: list[int], keepdim: bool) -> torch.Tensor:
    return torch.all(x, dim=dim[0] if len(dim) == 1 else None, keepdim=keepdim)


@tfl_reduce_all.register_fake
def _(x, dim, keepdim):
    return torch.all(x, dim=dim[0] if len(dim) == 1 else None, keepdim=keepdim)


# 141: CONV_3D_TRANSPOSE
# (Placeholder for 3D transpose convolution)


# 142: VAR_HANDLE
# (Placeholder for variable operation)


# 143: READ_VARIABLE
# (Placeholder for variable operation)


# 144: ASSIGN_VARIABLE
# (Placeholder for variable operation)


# 145: BROADCAST_ARGS
@torch.library.custom_op("tfl::broadcast_args", mutates_args=())
def tfl_broadcast_args(s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
    # Returns the shape resulting from broadcasting
    shape = torch.broadcast_shapes(tuple(s0.tolist()), tuple(s1.tolist()))
    return torch.tensor(list(shape))


@tfl_broadcast_args.register_fake
def _(s0, s1):
    shape = torch.broadcast_shapes(tuple(s0.tolist()), tuple(s1.tolist()))
    return torch.tensor(list(shape))


# 146: RANDOM_STANDARD_NORMAL
@torch.library.custom_op("tfl::random_standard_normal", mutates_args=())
def tfl_random_standard_normal(shape: list[int]) -> torch.Tensor:
    return torch.randn(shape)


@tfl_random_standard_normal.register_fake
def _(shape):
    return torch.randn(shape)


# 147: BUCKETIZE
@torch.library.custom_op("tfl::bucketize", mutates_args=())
def tfl_bucketize(x: torch.Tensor, boundaries: list[float]) -> torch.Tensor:
    return torch.bucketize(x, torch.tensor(boundaries))


@tfl_bucketize.register_fake
def _(x, boundaries):
    return torch.bucketize(x, torch.tensor(boundaries))


# 148: RANDOM_UNIFORM
@torch.library.custom_op("tfl::random_uniform", mutates_args=())
def tfl_random_uniform(shape: list[int], minval: float, maxval: float) -> torch.Tensor:
    return torch.rand(shape) * (maxval - minval) + minval


@tfl_random_uniform.register_fake
def _(shape, minval, maxval):
    return torch.rand(shape) * (maxval - minval) + minval


# 149: MULTINOMIAL
@torch.library.custom_op("tfl::multinomial", mutates_args=())
def tfl_multinomial(logits: torch.Tensor, num_samples: int) -> torch.Tensor:
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples)


@tfl_multinomial.register_fake
def _(logits, num_samples):
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples)


# 150: GELU
@torch.library.custom_op("tfl::gelu", mutates_args=())
def tfl_gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)


@tfl_gelu.register_fake
def _(x):
    return torch.nn.functional.gelu(x)


# 151: DYNAMIC_UPDATE_SLICE
@torch.library.custom_op("tfl::dynamic_update_slice", mutates_args=())
def tfl_dynamic_update_slice(operand: torch.Tensor, update: torch.Tensor, start_indices: list[int]) -> torch.Tensor:
    # Simplified implementation
    return operand


@tfl_dynamic_update_slice.register_fake
def _(operand, update, start_indices):
    return operand


# 152: RELU_0_TO_1
@torch.library.custom_op("tfl::relu_0_to_1", mutates_args=())
def tfl_relu_0_to_1(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0, max=1.0)


@tfl_relu_0_to_1.register_fake
def _(x):
    return torch.clamp(x, min=0.0, max=1.0)


# 153: UNSORTED_SEGMENT_PROD
@torch.library.custom_op("tfl::unsorted_segment_prod", mutates_args=())
def tfl_unsorted_segment_prod(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    # Simplified implementation
    return data


@tfl_unsorted_segment_prod.register_fake
def _(data, segment_ids, num_segments):
    return data


# 154: UNSORTED_SEGMENT_MAX
@torch.library.custom_op("tfl::unsorted_segment_max", mutates_args=())
def tfl_unsorted_segment_max(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    # Simplified implementation
    return data


@tfl_unsorted_segment_max.register_fake
def _(data, segment_ids, num_segments):
    return data


# 155: UNSORTED_SEGMENT_SUM
@torch.library.custom_op("tfl::unsorted_segment_sum", mutates_args=())
def tfl_unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    # Simplified implementation
    return data


@tfl_unsorted_segment_sum.register_fake
def _(data, segment_ids, num_segments):
    return data


# 156: ATAN2
@torch.library.custom_op("tfl::atan2", mutates_args=())
def tfl_atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(y, x)


@tfl_atan2.register_fake
def _(y, x):
    return torch.atan2(y, x)


# 157: UNSORTED_SEGMENT_MIN
@torch.library.custom_op("tfl::unsorted_segment_min", mutates_args=())
def tfl_unsorted_segment_min(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    # Simplified implementation
    return data


@tfl_unsorted_segment_min.register_fake
def _(data, segment_ids, num_segments):
    return data


# 158: SIGN
@torch.library.custom_op("tfl::sign", mutates_args=())
def tfl_sign(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x)


@tfl_sign.register_fake
def _(x):
    return torch.sign(x)


# 159: BITCAST
@torch.library.custom_op("tfl::bitcast", mutates_args=())
def tfl_bitcast(x: torch.Tensor, dtype: int) -> torch.Tensor:
    # Bitcast implementation
    return x


@tfl_bitcast.register_fake
def _(x, dtype):
    return x


# 160: BITWISE_XOR
@torch.library.custom_op("tfl::bitwise_xor", mutates_args=())
def tfl_bitwise_xor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_xor(x, y)


@tfl_bitwise_xor.register_fake
def _(x, y):
    return torch.bitwise_xor(x, y)


# 161: RIGHT_SHIFT
@torch.library.custom_op("tfl::right_shift", mutates_args=())
def tfl_right_shift(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_right_shift(x, y)


@tfl_right_shift.register_fake
def _(x, y):
    return torch.bitwise_right_shift(x, y)


# 203: DILATE
# (Placeholder for dilate operation)
