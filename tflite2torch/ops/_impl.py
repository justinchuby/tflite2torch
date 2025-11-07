"""Custom ops implementations for tflite ops in PyTorch.

This file implements custom ops for TFLite operators in the order they appear
in the TFLite BuiltinOperator enum (see plans/tflite_ops.md).
"""

import torch


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
def tfl_conv_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride_h: int = 1,
    stride_w: int = 1,
    padding: str = "SAME",
    fused_activation_function: str = "NONE",
) -> torch.Tensor:
    # TFLite CONV_2D with proper parameter handling
    # TFLite uses NHWC format, PyTorch uses NCHW
    # Input x: NHWC -> NCHW
    x = x.permute(0, 3, 1, 2) if x.dim() == 4 else x

    # Weight: TFLite format is [out_channels, kernel_h, kernel_w, in_channels]
    # PyTorch format is [out_channels, in_channels, kernel_h, kernel_w]
    weight = weight.permute(0, 3, 1, 2) if weight.dim() == 4 else weight

    stride = [stride_h, stride_w]

    # Handle padding - for now simplified (would need proper SAME/VALID calculation)
    pad = 0 if padding == "VALID" else 1

    # Apply convolution
    result = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=pad)

    # Apply fused activation
    if fused_activation_function == "RELU":
        result = torch.nn.functional.relu(result)
    elif fused_activation_function == "RELU6":
        result = torch.nn.functional.relu6(result)
    elif fused_activation_function == "TANH":
        result = torch.tanh(result)
    # NONE means no activation

    # Convert back to NHWC
    result = result.permute(0, 2, 3, 1)

    return result


@tfl_conv_2d.register_fake
def _(x, weight, bias, stride_h=1, stride_w=1, padding="SAME", fused_activation_function="NONE"):
    x = x.permute(0, 3, 1, 2) if x.dim() == 4 else x
    weight = weight.permute(0, 3, 1, 2) if weight.dim() == 4 else weight
    stride = [stride_h, stride_w]
    pad = 0 if padding == "VALID" else 1
    result = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=pad)
    result = result.permute(0, 2, 3, 1)
    return result


# 4: DEPTHWISE_CONV_2D
@torch.library.custom_op("tfl::depthwise_conv_2d", mutates_args=())
def tfl_depthwise_conv_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride_h: int = 1,
    stride_w: int = 1,
    padding: str = "SAME",
    fused_activation_function: str = "NONE",
) -> torch.Tensor:
    # TFLite DEPTHWISE_CONV_2D
    # Convert NHWC -> NCHW
    x = x.permute(0, 3, 1, 2) if x.dim() == 4 else x

    # TFLite depthwise weight format: [1, kernel_h, kernel_w, in_channels * multiplier]
    # PyTorch depthwise weight format: [in_channels * multiplier, 1, kernel_h, kernel_w]
    weight = weight.permute(3, 0, 1, 2) if weight.dim() == 4 else weight

    stride = [stride_h, stride_w]
    pad = 0 if padding == "VALID" else 1

    # Depthwise convolution uses groups = in_channels
    groups = x.shape[1]  # Channel dimension in NCHW
    result = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=pad, groups=groups)

    # Apply fused activation
    if fused_activation_function == "RELU":
        result = torch.nn.functional.relu(result)
    elif fused_activation_function == "RELU6":
        result = torch.nn.functional.relu6(result)
    elif fused_activation_function == "TANH":
        result = torch.tanh(result)

    # Convert back to NHWC
    result = result.permute(0, 2, 3, 1)

    return result


@tfl_depthwise_conv_2d.register_fake
def _(x, weight, bias, stride_h=1, stride_w=1, padding="SAME", fused_activation_function="NONE"):
    x = x.permute(0, 3, 1, 2) if x.dim() == 4 else x
    weight = weight.permute(3, 0, 1, 2) if weight.dim() == 4 else weight
    stride = [stride_h, stride_w]
    pad = 0 if padding == "VALID" else 1
    groups = x.shape[1]
    result = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=pad, groups=groups)
    result = result.permute(0, 2, 3, 1)
    return result


# 5: DEPTH_TO_SPACE
@torch.library.custom_op("tfl::depth_to_space", mutates_args=())
def tfl_depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.nn.functional.pixel_shuffle(x, block_size)


@tfl_depth_to_space.register_fake
def _(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


# 6: DEQUANTIZE
@torch.library.custom_op("tfl::dequantize", mutates_args=())
def tfl_dequantize(x: torch.Tensor, scale: float = 1.0, zero_point: int = 0) -> torch.Tensor:
    # Dequantize: convert quantized int8/uint8 to float
    # real_value = scale * (quantized_value - zero_point)
    return scale * (x.float() - zero_point)


@tfl_dequantize.register_fake
def _(x, scale=1.0, zero_point=0):
    return scale * (x.float() - zero_point)


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
def tfl_fully_connected(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    fused_activation_function: str = "NONE",
) -> torch.Tensor:
    result = torch.nn.functional.linear(x, weight, bias)

    # Apply fused activation
    if fused_activation_function == "RELU":
        result = torch.nn.functional.relu(result)
    elif fused_activation_function == "RELU6":
        result = torch.nn.functional.relu6(result)
    elif fused_activation_function == "TANH":
        result = torch.tanh(result)
    # NONE means no activation

    return result


@tfl_fully_connected.register_fake
def _(x, weight, bias, fused_activation_function="NONE"):
    return torch.nn.functional.linear(x, weight, bias)


# 10: HASHTABLE_LOOKUP
@torch.library.custom_op("tfl::hashtable_lookup", mutates_args=())
def tfl_hashtable_lookup(keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    # Lookup values from a hashtable given query keys
    # Simple implementation: for each query, find matching key and return corresponding value
    result_list = []
    for q in query.flatten():
        # Find index where key matches query
        matches = (keys == q).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            result_list.append(values[matches[0]])
        else:
            # Return zero/default if not found
            result_list.append(torch.zeros_like(values[0]))

    result = torch.stack(result_list)
    return result.reshape(query.shape + values.shape[1:])


@tfl_hashtable_lookup.register_fake
def _(keys, values, query):
    output_shape = list(query.shape) + list(values.shape[1:])
    return torch.empty(output_shape, dtype=values.dtype)


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
    return x.clone()


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
def tfl_reshape(x: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    # Convert tensor shape parameter to list
    shape_list = shape.tolist() if isinstance(shape, torch.Tensor) else shape
    # Convert floats to ints if needed
    shape_list = [int(s) for s in shape_list]
    # Use reshape which returns a clone to avoid aliasing
    result = torch.reshape(x, shape_list)
    # Clone to avoid aliasing issues with custom ops
    return result.clone()


@tfl_reshape.register_fake
def _(x, shape):
    shape_list = shape.tolist() if isinstance(shape, torch.Tensor) else shape
    shape_list = [int(s) for s in shape_list]
    return torch.empty(shape_list, dtype=x.dtype)


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
    # Sparse embedding lookup with weighted aggregation
    # Look up embeddings for indices and multiply by weights
    embeddings = torch.nn.functional.embedding(indices, params)
    # Apply weights if provided
    if weights is not None and weights.numel() > 0:
        # Broadcast weights to match embedding dimensions
        weights_expanded = weights.unsqueeze(-1)
        embeddings = embeddings * weights_expanded
    return embeddings


@tfl_embedding_lookup_sparse.register_fake
def _(params, indices, weights):
    embedding_dim = params.shape[-1]
    output_shape = list(indices.shape) + [embedding_dim]
    return torch.empty(output_shape, dtype=params.dtype)


# 34: PAD
@torch.library.custom_op("tfl::pad", mutates_args=())
def tfl_pad(x: torch.Tensor, paddings: torch.Tensor) -> torch.Tensor:
    # TFLite padding format: [[dim0_before, dim0_after], [dim1_before, dim1_after], ...]
    # PyTorch pad format: (dimN_before, dimN_after, ..., dim1_before, dim1_after, dim0_before, dim0_after)
    # Need to reverse the order and flatten

    # Reshape paddings if needed
    if paddings.ndim == 2:
        # Standard format: [[before, after], ...]
        pad_pairs = paddings.tolist()
    else:
        # Already flattened
        pad_flat = paddings.flatten().tolist()
        pad_pairs = [[int(pad_flat[i]), int(pad_flat[i+1])] for i in range(0, len(pad_flat), 2)]

    # Reverse the order for PyTorch (last dim first)
    pad_list = []
    for before, after in reversed(pad_pairs):
        pad_list.extend([int(before), int(after)])

    result = torch.nn.functional.pad(x, pad_list)
    return result.clone()


@tfl_pad.register_fake
def _(x, paddings):
    if paddings.ndim == 2:
        pad_pairs = paddings.tolist()
    else:
        pad_flat = paddings.flatten().tolist()
        pad_pairs = [[int(pad_flat[i]), int(pad_flat[i+1])] for i in range(0, len(pad_flat), 2)]

    pad_list = []
    for before, after in reversed(pad_pairs):
        pad_list.extend([int(before), int(after)])

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
def tfl_gather(params: torch.Tensor, indices: torch.Tensor, axis: int = 0) -> torch.Tensor:
    # TFLite gather - select values from params using indices along axis
    # Ensure indices are int64
    indices_int = indices.long() if indices.dtype != torch.int64 else indices
    result = torch.index_select(params, axis, indices_int.flatten())
    out_shape = list(params.shape[:axis]) + list(indices.shape) + list(params.shape[axis+1:])
    return result.reshape(out_shape).clone()


@tfl_gather.register_fake
def _(params, indices, axis=0):
    out_shape = list(params.shape[:axis]) + list(indices.shape) + list(params.shape[axis+1:])
    return torch.empty(out_shape, dtype=params.dtype)


# 37: BATCH_TO_SPACE_ND
@torch.library.custom_op("tfl::batch_to_space_nd", mutates_args=())
def tfl_batch_to_space_nd(x: torch.Tensor, block_shape: list[int], crops: list[int]) -> torch.Tensor:
    # Rearranges data from batch into blocks of spatial data
    # This is the inverse of SPACE_TO_BATCH_ND
    # crops is a flattened list: [crop_top, crop_bottom, crop_left, crop_right]
    batch, height, width, channels = x.shape
    block_height, block_width = block_shape

    # Reshape to extract blocks
    x = x.reshape(batch // (block_height * block_width), block_height, block_width, height, width, channels)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(batch // (block_height * block_width), height * block_height, width * block_width, channels)

    # Apply crops - crops format: [crop_top, crop_bottom, crop_left, crop_right]
    if crops and any(c != 0 for c in crops):
        crop_top, crop_bottom, crop_left, crop_right = crops[0], crops[1], crops[2], crops[3]
        x = x[:, crop_top:height * block_height - crop_bottom, crop_left:width * block_width - crop_right, :]

    return x


@tfl_batch_to_space_nd.register_fake
def _(x, block_shape, crops):
    batch, height, width, channels = x.shape
    block_height, block_width = block_shape
    new_height = height * block_height
    new_width = width * block_width
    if crops and any(c != [0, 0] for c in crops):
        crop_top, crop_bottom = crops[0]
        crop_left, crop_right = crops[1]
        new_height = new_height - crop_top - crop_bottom
        new_width = new_width - crop_left - crop_right
    return torch.empty(batch // (block_height * block_width), new_height, new_width, channels)


# 38: SPACE_TO_BATCH_ND
@torch.library.custom_op("tfl::space_to_batch_nd", mutates_args=())
def tfl_space_to_batch_nd(x: torch.Tensor, block_shape: list[int], paddings: list[int]) -> torch.Tensor:
    # Rearranges blocks of spatial data into batch
    # paddings is a flattened list: [pad_top, pad_bottom, pad_left, pad_right]
    batch, height, width, channels = x.shape
    block_height, block_width = block_shape

    # Apply padding - paddings format: [pad_top, pad_bottom, pad_left, pad_right]
    if paddings and any(p != 0 for p in paddings):
        pad_top, pad_bottom, pad_left, pad_right = paddings[0], paddings[1], paddings[2], paddings[3]
        x = torch.nn.functional.pad(x.permute(0, 3, 1, 2), (pad_left, pad_right, pad_top, pad_bottom)).permute(0, 2, 3, 1)
        height = height + pad_top + pad_bottom
        width = width + pad_left + pad_right

    # Reshape to create blocks
    x = x.reshape(batch, height // block_height, block_height, width // block_width, block_width, channels)
    x = x.permute(2, 4, 0, 1, 3, 5)
    x = x.reshape(batch * block_height * block_width, height // block_height, width // block_width, channels)

    return x


@tfl_space_to_batch_nd.register_fake
def _(x, block_shape, paddings):
    batch, height, width, channels = x.shape
    block_height, block_width = block_shape
    if paddings and any(p != [0, 0] for p in paddings):
        pad_top, pad_bottom = paddings[0]
        pad_left, pad_right = paddings[1]
        height = height + pad_top + pad_bottom
        width = width + pad_left + pad_right
    return torch.empty(batch * block_height * block_width, height // block_height, width // block_width, channels)


# 39: TRANSPOSE
@torch.library.custom_op("tfl::transpose", mutates_args=())
def tfl_transpose(x: torch.Tensor, perm: list[int]) -> torch.Tensor:
    return torch.permute(x, perm)


@tfl_transpose.register_fake
def _(x, perm):
    return torch.permute(x, perm)


# 40: MEAN
@torch.library.custom_op("tfl::mean", mutates_args=())
def tfl_mean(x: torch.Tensor, dim: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    # Convert tensor dim parameter to list/int
    if isinstance(dim, torch.Tensor):
        dim_list = dim.tolist()
        # Convert to int list
        dim_list = [int(d) for d in dim_list] if isinstance(dim_list, list) else int(dim_list)
    else:
        dim_list = dim
    return torch.mean(x, dim=dim_list, keepdim=keepdim)


@tfl_mean.register_fake
def _(x, dim, keepdim=False):
    if isinstance(dim, torch.Tensor):
        dim_list = dim.tolist()
        dim_list = [int(d) for d in dim_list] if isinstance(dim_list, list) else int(dim_list)
    else:
        dim_list = dim
    return torch.mean(x, dim=dim_list, keepdim=keepdim)


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
def tfl_strided_slice(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    # TFLite STRIDED_SLICE: extract a strided slice from a tensor
    # Convert tensor parameters to lists
    begin_list = begin.tolist() if isinstance(begin, torch.Tensor) else begin
    end_list = end.tolist() if isinstance(end, torch.Tensor) else end
    strides_list = strides.tolist() if isinstance(strides, torch.Tensor) else strides

    slices = []
    for b, e, s in zip(begin_list, end_list, strides_list):
        slices.append(slice(int(b), int(e), int(s)))
    # Clone to avoid aliasing issues with custom ops
    return x[tuple(slices)].clone()


@tfl_strided_slice.register_fake
def _(x, begin, end, strides):
    # For shape inference, just return same shape (simplified)
    return torch.empty_like(x)


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
# Note: TFLite SPLIT with num_splits=1 returns a single tensor (identity operation)
# For multiple splits, TFLite typically uses SPLIT_V instead
@torch.library.custom_op("tfl::split", mutates_args=())
def tfl_split(split_dim: torch.Tensor, x: torch.Tensor, num_splits: int = 1) -> torch.Tensor:
    # TFLite SPLIT: split_dim is a scalar tensor indicating dimension to split
    dim = int(split_dim.item()) if isinstance(split_dim, torch.Tensor) else split_dim
    # When num_splits is 1, return the tensor as-is (identity operation in TFLite)
    if num_splits == 1:
        return x.clone()
    # When num_splits > 1, return the first split (this is a fallback - normally SPLIT_V is used)
    # Note: This signature can only return one tensor, not a list
    splits = torch.split(x, x.shape[dim] // num_splits, dim=dim)
    return splits[0].clone()


@tfl_split.register_fake
def _(split_dim, x, num_splits=1):
    # Return the input shape unchanged when num_splits=1
    if num_splits == 1:
        return x
    # When num_splits > 1, return the shape of first split
    dim = int(split_dim.item()) if isinstance(split_dim, torch.Tensor) else split_dim
    shape = list(x.shape)
    shape[dim] = shape[dim] // num_splits
    return torch.empty(shape, dtype=x.dtype, device=x.device)


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
    # TFLite dtype enum to PyTorch dtype mapping
    # Common TFLite types: FLOAT32=0, INT32=2, UINT8=3, INT64=4, BOOL=6, INT16=7, FLOAT16=10
    dtype_map = {
        0: torch.float32,   # FLOAT32
        1: torch.float16,   # FLOAT16
        2: torch.int32,     # INT32
        3: torch.uint8,     # UINT8
        4: torch.int64,     # INT64
        # 5: STRING (not supported in PyTorch tensors)
        6: torch.bool,      # BOOL
        7: torch.int16,     # INT16
        8: torch.complex64, # COMPLEX64
        9: torch.int8,      # INT8
        10: torch.float16,  # FLOAT16
    }

    target_dtype = dtype_map.get(dtype, x.dtype)
    return x.to(target_dtype)


@tfl_cast.register_fake
def _(x, dtype):
    dtype_map = {
        0: torch.float32, 1: torch.float16, 2: torch.int32, 3: torch.uint8,
        4: torch.int64, 6: torch.bool, 7: torch.int16, 8: torch.complex64,
        9: torch.int8, 10: torch.float16,
    }
    target_dtype = dtype_map.get(dtype, x.dtype)
    return x.to(target_dtype)


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
def tfl_slice(x: torch.Tensor, begin: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    # TFLite SLICE: extract a slice from a tensor
    # begin: starting indices for each dimension
    # size: size of the slice for each dimension (-1 means remaining)
    begin_list = begin.tolist() if isinstance(begin, torch.Tensor) else begin
    size_list = size.tolist() if isinstance(size, torch.Tensor) else size

    slices = []
    for i, (b, s) in enumerate(zip(begin_list, size_list)):
        b, s = int(b), int(s)
        if s == -1:
            slices.append(slice(b, None))
        else:
            slices.append(slice(b, b + s))
    # Clone to avoid aliasing
    return x[tuple(slices)].clone()


@tfl_slice.register_fake
def _(x, begin, size):
    return torch.empty_like(x)


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
    # Convert sparse representation to dense tensor
    output = torch.full(output_shape, default_value, dtype=sparse_values.dtype)

    # Fill in sparse values at specified indices
    if sparse_indices.ndim == 1:
        # 1D indices
        for i, idx in enumerate(sparse_indices):
            output[int(idx)] = sparse_values[i]
    else:
        # Multi-dimensional indices
        for i in range(sparse_indices.shape[0]):
            idx = tuple(int(sparse_indices[i, j]) for j in range(sparse_indices.shape[1]))
            output[idx] = sparse_values[i]

    return output


@tfl_sparse_to_dense.register_fake
def _(sparse_indices, output_shape, sparse_values, default_value):
    return torch.full(output_shape, default_value, dtype=sparse_values.dtype)


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
def tfl_reduce_prod(x: torch.Tensor, dim: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    # Convert dim tensor to list of ints
    if dim.numel() == 0:
        # Scalar tensor - single dimension
        return torch.prod(x)
    elif dim.dim() == 0:
        # 0-dimensional tensor (scalar)
        dim_list = [int(dim.item())]
    else:
        dim_list = dim.tolist() if isinstance(dim.tolist(), list) else [int(dim.item())]

    result = x
    for d in sorted(dim_list, reverse=True):
        result = torch.prod(result, dim=d, keepdim=keepdim)
    return result


@tfl_reduce_prod.register_fake
def _(x, dim, keepdim=False):
    if dim.numel() == 0:
        return torch.prod(x)
    elif dim.dim() == 0:
        dim_list = [int(dim.item())]
    else:
        dim_list = dim.tolist() if isinstance(dim.tolist(), list) else [int(dim.item())]

    result = x
    for d in sorted(dim_list, reverse=True):
        result = torch.prod(result, dim=d, keepdim=keepdim)
    return result


# 82: REDUCE_MAX
@torch.library.custom_op("tfl::reduce_max", mutates_args=())
def tfl_reduce_max(x: torch.Tensor, dim: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    if dim.numel() == 0:
        return torch.amax(x)
    elif dim.dim() == 0:
        dim_list = [int(dim.item())]
    else:
        dim_list = dim.tolist() if isinstance(dim.tolist(), list) else [int(dim.item())]

    result = x
    for d in sorted(dim_list, reverse=True):
        result = torch.amax(result, dim=d, keepdim=keepdim)
    return result


@tfl_reduce_max.register_fake
def _(x, dim, keepdim=False):
    if dim.numel() == 0:
        return torch.amax(x)
    elif dim.dim() == 0:
        dim_list = [int(dim.item())]
    else:
        dim_list = dim.tolist() if isinstance(dim.tolist(), list) else [int(dim.item())]

    result = x
    for d in sorted(dim_list, reverse=True):
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
def tfl_reduce_min(x: torch.Tensor, dim: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    if dim.numel() == 0:
        return torch.amin(x)
    elif dim.dim() == 0:
        dim_list = [int(dim.item())]
    else:
        dim_list = dim.tolist() if isinstance(dim.tolist(), list) else [int(dim.item())]

    result = x
    for d in sorted(dim_list, reverse=True):
        result = torch.amin(result, dim=d, keepdim=keepdim)
    return result


@tfl_reduce_min.register_fake
def _(x, dim, keepdim=False):
    if dim.numel() == 0:
        return torch.amin(x)
    elif dim.dim() == 0:
        dim_list = [int(dim.item())]
    else:
        dim_list = dim.tolist() if isinstance(dim.tolist(), list) else [int(dim.item())]

    result = x
    for d in sorted(dim_list, reverse=True):
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
def tfl_reduce_any(x: torch.Tensor, dim: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    if dim.numel() == 0:
        return torch.any(x)
    elif dim.dim() == 0:
        dim_list = [int(dim.item())]
    else:
        dim_list = dim.tolist() if isinstance(dim.tolist(), list) else [int(dim.item())]

    result = x
    for d in sorted(dim_list, reverse=True):
        result = torch.any(result, dim=d, keepdim=keepdim)
    return result


@tfl_reduce_any.register_fake
def _(x, dim, keepdim=False):
    if dim.numel() == 0:
        return torch.any(x)
    elif dim.dim() == 0:
        dim_list = [int(dim.item())]
    else:
        dim_list = dim.tolist() if isinstance(dim.tolist(), list) else [int(dim.item())]

    result = x
    for d in sorted(dim_list, reverse=True):
        result = torch.any(result, dim=d, keepdim=keepdim)
    return result


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
    # Convert TFLite paddings format [[top, bottom], [left, right], ...] to PyTorch format
    # PyTorch pad format is (left, right, top, bottom) for 2D
    pad_list = []
    # Reverse order and flatten for PyTorch
    for i in range(paddings.shape[0] - 1, -1, -1):
        pad_list.extend([int(paddings[i, 0]), int(paddings[i, 1])])

    # Map mode: 'REFLECT' or 'SYMMETRIC'
    torch_mode = 'reflect' if mode == 'REFLECT' else 'replicate'
    return torch.nn.functional.pad(x, pad_list, mode=torch_mode)


@tfl_mirror_pad.register_fake
def _(x, paddings, mode):
    pad_list = []
    for i in range(paddings.shape[0] - 1, -1, -1):
        pad_list.extend([int(paddings[i, 0]), int(paddings[i, 1])])
    torch_mode = 'reflect' if mode == 'REFLECT' else 'replicate'
    return torch.nn.functional.pad(x, pad_list, mode=torch_mode)


# 101: ABS
@torch.library.custom_op("tfl::abs", mutates_args=())
def tfl_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


@tfl_abs.register_fake
def _(x):
    return torch.abs(x)


# 102: SPLIT_V
@torch.library.custom_op("tfl::split_v", mutates_args=())
def tfl_split_v(x: torch.Tensor, size_splits: torch.Tensor, dim: torch.Tensor) -> list[torch.Tensor]:
    # Convert size_splits tensor to list of ints
    if size_splits.dim() == 0:
        splits_list = [int(size_splits.item())]
    else:
        splits_list = [int(s) for s in size_splits.tolist()]

    # Convert dim to int
    dim_int = int(dim.item()) if isinstance(dim, torch.Tensor) else dim

    # Handle splits - PyTorch split_with_sizes handles zero-size tensors
    # Filter negative values (TFLite uses -1 for "infer remaining")
    total_size = x.shape[dim_int]
    inferred_idx = -1
    inferred_size = total_size

    for i, s in enumerate(splits_list):
        if s == -1:
            inferred_idx = i
        elif s > 0:
            inferred_size -= s

    # Replace -1 with inferred size
    if inferred_idx >= 0:
        splits_list[inferred_idx] = inferred_size

    # Use split with sizes
    return [t.clone() for t in torch.split(x, splits_list, dim=dim_int)]


@tfl_split_v.register_fake
def _(x, size_splits, dim):
    if size_splits.dim() == 0:
        splits_list = [int(size_splits.item())]
    else:
        splits_list = [int(s) for s in size_splits.tolist()]

    dim_int = int(dim.item()) if isinstance(dim, torch.Tensor) else dim

    total_size = x.shape[dim_int]
    inferred_idx = -1
    inferred_size = total_size

    for i, s in enumerate(splits_list):
        if s == -1:
            inferred_idx = i
        elif s > 0:
            inferred_size -= s

    if inferred_idx >= 0:
        splits_list[inferred_idx] = inferred_size

    return list(torch.split(x, splits_list, dim=dim_int))
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
    # GATHER_ND gathers slices from params using N-dimensional indices
    # indices shape: [..., M] where M <= params.ndim
    # For each index tuple in indices, gather the corresponding element from params
    indices_shape = indices.shape
    indices = indices.reshape(-1, indices_shape[-1])

    result = []
    for idx in indices:
        val = params
        for i in idx:
            val = val[i]
        result.append(val)

    result = torch.stack(result)
    new_shape = list(indices_shape[:-1]) + list(result.shape[1:])
    return result.reshape(new_shape)


@tfl_gather_nd.register_fake
def _(params, indices):
    indices_shape = indices.shape
    num_indices = indices_shape[-1]
    result_shape = list(indices_shape[:-1]) + list(params.shape[num_indices:])
    return torch.empty(result_shape, dtype=params.dtype)


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
    # Reverses variable length slices in specified dimension
    output = x.clone()

    for i in range(x.shape[batch_dim]):
        seq_len = int(seq_lengths[i])

        # Handle common case first
        if batch_dim == 0 and seq_dim == 1:
            output[i, :seq_len] = torch.flip(x[i, :seq_len], dims=[0])
        elif batch_dim == 0 and seq_dim == 0:
            # Can't reverse batch dimension itself
            continue
        else:
            # For other cases, use movedim to bring batch_dim to front
            x_moved = x.movedim(batch_dim, 0)
            out_moved = output.movedim(batch_dim, 0)

            # Get the sequence for this batch element
            seq = x_moved[i]
            # Adjust seq_dim for moved tensor
            adj_seq_dim = seq_dim if seq_dim < batch_dim else seq_dim - 1

            # Create slice for reversing
            slices = [slice(None)] * seq.ndim
            slices[adj_seq_dim] = slice(0, seq_len)
            out_moved[i][tuple(slices)] = torch.flip(seq[tuple(slices)], dims=[adj_seq_dim])

    return output


@tfl_reverse_sequence.register_fake
def _(x, seq_lengths, seq_dim, batch_dim):
    return x.clone()


# 113: MATRIX_DIAG
@torch.library.custom_op("tfl::matrix_diag", mutates_args=())
def tfl_matrix_diag(diagonal: torch.Tensor) -> torch.Tensor:
    return torch.diag_embed(diagonal)


@tfl_matrix_diag.register_fake
def _(diagonal):
    return torch.diag_embed(diagonal)


# 114: QUANTIZE
@torch.library.custom_op("tfl::quantize", mutates_args=())
def tfl_quantize(x: torch.Tensor) -> torch.Tensor:
    # Simplified quantize - in TFLite this uses quantization parameters from the model
    # For now, just return a clone (quantization info is in the model metadata)
    return x.clone()


@tfl_quantize.register_fake
def _(x):
    return x


# 115: MATRIX_SET_DIAG
@torch.library.custom_op("tfl::matrix_set_diag", mutates_args=())
def tfl_matrix_set_diag(x: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    # Sets the diagonal of a matrix to the given values
    output = x.clone()
    if x.ndim == 2:
        # Single matrix
        min_dim = min(x.shape[0], x.shape[1])
        for i in range(min_dim):
            output[i, i] = diagonal[i]
    else:
        # Batched matrices
        batch_shape = x.shape[:-2]
        min_dim = min(x.shape[-2], x.shape[-1])
        import itertools
        for idx in itertools.product(*[range(s) for s in batch_shape]):
            for i in range(min_dim):
                output[idx + (i, i)] = diagonal[idx + (i,)]
    return output


@tfl_matrix_set_diag.register_fake
def _(x, diagonal):
    return x.clone()


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
@torch.library.custom_op("tfl::if_op", mutates_args=())
def tfl_if_op(cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Conditional operation - returns x if cond is true, else y
    return torch.where(cond, x, y)


@tfl_if_op.register_fake
def _(cond, x, y):
    return torch.where(cond, x, y)


# 119: WHILE
@torch.library.custom_op("tfl::while_op", mutates_args=())
def tfl_while_op(cond: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Simplified while loop - in practice this requires control flow graphs
    # For now, just return the input as control flow needs special handling
    return x


@tfl_while_op.register_fake
def _(cond, x):
    return x


# 120: NON_MAX_SUPPRESSION_V4
@torch.library.custom_op("tfl::non_max_suppression_v4", mutates_args=())
def tfl_non_max_suppression_v4(boxes: torch.Tensor, scores: torch.Tensor, max_output_size: int,
                                iou_threshold: float, score_threshold: float) -> torch.Tensor:
    # Non-maximum suppression for object detection
    # Filter by score threshold
    valid_mask = scores >= score_threshold
    valid_indices = torch.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return torch.empty(0, dtype=torch.int64)

    # Use torchvision-style NMS
    from torchvision.ops import nms
    selected = nms(boxes[valid_indices], scores[valid_indices], iou_threshold)

    # Limit to max output size
    if len(selected) > max_output_size:
        selected = selected[:max_output_size]

    return valid_indices[selected]


@tfl_non_max_suppression_v4.register_fake
def _(boxes, scores, max_output_size, iou_threshold, score_threshold):
    return torch.empty(min(max_output_size, len(scores)), dtype=torch.int64)


# 121: NON_MAX_SUPPRESSION_V5
@torch.library.custom_op("tfl::non_max_suppression_v5", mutates_args=())
def tfl_non_max_suppression_v5(boxes: torch.Tensor, scores: torch.Tensor, max_output_size: int,
                                iou_threshold: float, score_threshold: float, soft_nms_sigma: float) -> torch.Tensor:
    # NMS V5 with soft-NMS support (simplified - using hard NMS)
    valid_mask = scores >= score_threshold
    valid_indices = torch.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return torch.empty(0, dtype=torch.int64)

    from torchvision.ops import nms
    selected = nms(boxes[valid_indices], scores[valid_indices], iou_threshold)

    if len(selected) > max_output_size:
        selected = selected[:max_output_size]

    return valid_indices[selected]


@tfl_non_max_suppression_v5.register_fake
def _(boxes, scores, max_output_size, iou_threshold, score_threshold, soft_nms_sigma):
    return torch.empty(min(max_output_size, len(scores)), dtype=torch.int64)


# 122: SCATTER_ND
@torch.library.custom_op("tfl::scatter_nd", mutates_args=())
def tfl_scatter_nd(indices: torch.Tensor, updates: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    # Creates a tensor by scattering updates at indices
    shape_list = shape.tolist() if isinstance(shape, torch.Tensor) else shape
    shape_list = [int(s) for s in shape_list]
    output = torch.zeros(shape_list, dtype=updates.dtype)

    # Flatten indices for iteration
    indices_shape = indices.shape
    indices_flat = indices.reshape(-1, indices_shape[-1])
    updates_flat = updates.reshape(-1, *updates.shape[len(indices_shape)-1:])

    for i, idx in enumerate(indices_flat):
        idx_tuple = tuple(idx.tolist())
        output[idx_tuple] = updates_flat[i]

    return output


@tfl_scatter_nd.register_fake
def _(indices, updates, shape):
    return torch.zeros(shape, dtype=updates.dtype)


# 123: SELECT_V2
@torch.library.custom_op("tfl::select_v2", mutates_args=())
def tfl_select_v2(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(condition, x, y)


@tfl_select_v2.register_fake
def _(condition, x, y):
    return torch.where(condition, x, y)


# 124: DENSIFY
@torch.library.custom_op("tfl::densify", mutates_args=())
def tfl_densify(x: torch.Tensor) -> torch.Tensor:
    # Convert sparse tensor representation to dense
    # If x is already dense, return as-is
    if x.is_sparse:
        return x.to_dense()
    return x


@tfl_densify.register_fake
def _(x):
    return torch.empty_like(x)


# 125: SEGMENT_SUM
@torch.library.custom_op("tfl::segment_sum", mutates_args=())
def tfl_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    # Computes the sum along segments of a tensor
    num_segments = int(segment_ids.max()) + 1
    output_shape = [num_segments] + list(data.shape[1:])
    output = torch.zeros(output_shape, dtype=data.dtype)

    for i in range(len(segment_ids)):
        seg_id = int(segment_ids[i])
        output[seg_id] += data[i]

    return output


@tfl_segment_sum.register_fake
def _(data, segment_ids):
    num_segments = int(segment_ids.max()) + 1
    output_shape = [num_segments] + list(data.shape[1:])
    return torch.zeros(output_shape, dtype=data.dtype)


# 126: BATCH_MATMUL
@torch.library.custom_op("tfl::batch_matmul", mutates_args=())
def tfl_batch_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Batch matrix multiplication
    return torch.bmm(x, y)


@tfl_batch_matmul.register_fake
def _(x, y):
    return torch.bmm(x, y)


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
@torch.library.custom_op("tfl::call_once", mutates_args=())
def tfl_call_once(x: torch.Tensor) -> torch.Tensor:
    # Call once is for initialization functions - in inference mode, just pass through
    return x


@tfl_call_once.register_fake
def _(x):
    return x


# 130: BROADCAST_TO
@torch.library.custom_op("tfl::broadcast_to", mutates_args=())
def tfl_broadcast_to(x: torch.Tensor, shape: list[int]) -> torch.Tensor:
    return torch.broadcast_to(x, shape)


@tfl_broadcast_to.register_fake
def _(x, shape):
    return torch.broadcast_to(x, shape)


# 131: RFFT2D
@torch.library.custom_op("tfl::rfft2d", mutates_args=())
def tfl_rfft2d(x: torch.Tensor, fft_length: torch.Tensor) -> torch.Tensor:
    # Real-valued 2D FFT
    # fft_length parameter is typically ignored for now, using input shape
    return torch.fft.rfft2(x)


@tfl_rfft2d.register_fake
def _(x, fft_length):
    # Output shape: [..., H, W//2 + 1] with complex dtype
    shape = list(x.shape)
    shape[-1] = shape[-1] // 2 + 1
    return torch.empty(shape, dtype=torch.complex64)


# 132: CONV_3D
@torch.library.custom_op("tfl::conv_3d", mutates_args=())
def tfl_conv_3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: list[int], padding: str) -> torch.Tensor:
    # 3D convolution
    return torch.nn.functional.conv3d(x, weight, bias, stride)


@tfl_conv_3d.register_fake
def _(x, weight, bias, stride, padding):
    return torch.nn.functional.conv3d(x, weight, bias, stride)


# 133: IMAG
@torch.library.custom_op("tfl::imag", mutates_args=())
def tfl_imag(x: torch.Tensor) -> torch.Tensor:
    # Extract imaginary part of complex tensor
    return torch.imag(x)


@tfl_imag.register_fake
def _(x):
    return torch.imag(x)


# 134: REAL
@torch.library.custom_op("tfl::real", mutates_args=())
def tfl_real(x: torch.Tensor) -> torch.Tensor:
    # Extract real part of complex tensor
    return torch.real(x)


@tfl_real.register_fake
def _(x):
    return torch.real(x)


# 135: COMPLEX_ABS
@torch.library.custom_op("tfl::complex_abs", mutates_args=())
def tfl_complex_abs(x: torch.Tensor) -> torch.Tensor:
    # Absolute value (magnitude) of complex tensor
    return torch.abs(x)


@tfl_complex_abs.register_fake
def _(x):
    return torch.abs(x)


# 136: HASHTABLE
@torch.library.custom_op("tfl::hashtable", mutates_args=())
def tfl_hashtable(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # Create a hashtable representation - return a tensor with the table size
    return torch.tensor([len(keys)], dtype=torch.int64)


@tfl_hashtable.register_fake
def _(keys, values):
    return torch.tensor([len(keys)], dtype=torch.int64)


# 137: HASHTABLE_FIND
@torch.library.custom_op("tfl::hashtable_find", mutates_args=())
def tfl_hashtable_find(keys: torch.Tensor, values: torch.Tensor, query_keys: torch.Tensor, default_value: torch.Tensor) -> torch.Tensor:
    # Find values in hashtable by keys
    result = torch.full((len(query_keys),) + values.shape[1:], default_value.item() if default_value.numel() == 1 else 0, dtype=values.dtype)

    for i, query_key in enumerate(query_keys):
        matches = (keys == query_key).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            result[i] = values[matches[0]]

    return result


@tfl_hashtable_find.register_fake
def _(keys, values, query_keys, default_value):
    return torch.empty((len(query_keys),) + values.shape[1:], dtype=values.dtype)


# 138: HASHTABLE_IMPORT
@torch.library.custom_op("tfl::hashtable_import", mutates_args=())
def tfl_hashtable_import(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # Import data into hashtable - return a dummy tensor as hashtables aren't first-class
    return torch.tensor([len(keys)], dtype=torch.int64)


@tfl_hashtable_import.register_fake
def _(keys, values):
    return torch.tensor([len(keys)], dtype=torch.int64)


# 139: HASHTABLE_SIZE
@torch.library.custom_op("tfl::hashtable_size", mutates_args=())
def tfl_hashtable_size(keys: torch.Tensor) -> torch.Tensor:
    # Return the size of the hashtable
    return torch.tensor([len(keys)], dtype=torch.int64)


@tfl_hashtable_size.register_fake
def _(keys):
    return torch.tensor([len(keys)], dtype=torch.int64)


# 140: REDUCE_ALL
@torch.library.custom_op("tfl::reduce_all", mutates_args=())
def tfl_reduce_all(x: torch.Tensor, dim: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    if dim.numel() == 0:
        return torch.all(x)
    elif dim.dim() == 0:
        dim_list = [int(dim.item())]
    else:
        dim_list = dim.tolist() if isinstance(dim.tolist(), list) else [int(dim.item())]

    result = x
    for d in sorted(dim_list, reverse=True):
        result = torch.all(result, dim=d, keepdim=keepdim)
    return result


@tfl_reduce_all.register_fake
def _(x, dim, keepdim=False):
    if dim.numel() == 0:
        return torch.all(x)
    elif dim.dim() == 0:
        dim_list = [int(dim.item())]
    else:
        dim_list = dim.tolist() if isinstance(dim.tolist(), list) else [int(dim.item())]

    result = x
    for d in sorted(dim_list, reverse=True):
        result = torch.all(result, dim=d, keepdim=keepdim)
    return result


# 141: CONV_3D_TRANSPOSE
@torch.library.custom_op("tfl::conv_3d_transpose", mutates_args=())
def tfl_conv_3d_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, output_shape: list[int], stride: list[int]) -> torch.Tensor:
    # 3D transpose convolution
    return torch.nn.functional.conv_transpose3d(x, weight, bias, stride)


@tfl_conv_3d_transpose.register_fake
def _(x, weight, bias, output_shape, stride):
    return torch.nn.functional.conv_transpose3d(x, weight, bias, stride)


# 142: VAR_HANDLE
@torch.library.custom_op("tfl::var_handle", mutates_args=())
def tfl_var_handle(shape: list[int], dtype: int) -> torch.Tensor:
    # Create a variable handle - return empty tensor of specified shape/dtype
    dtype_map = {0: torch.float32, 2: torch.int32, 4: torch.int64}
    target_dtype = dtype_map.get(dtype, torch.float32)
    return torch.empty(shape, dtype=target_dtype)


@tfl_var_handle.register_fake
def _(shape, dtype):
    dtype_map = {0: torch.float32, 2: torch.int32, 4: torch.int64}
    target_dtype = dtype_map.get(dtype, torch.float32)
    return torch.empty(shape, dtype=target_dtype)


# 143: READ_VARIABLE
@torch.library.custom_op("tfl::read_variable", mutates_args=())
def tfl_read_variable(resource: torch.Tensor) -> torch.Tensor:
    # Read from a variable resource - just return the tensor
    return resource


@tfl_read_variable.register_fake
def _(resource):
    return resource


# 144: ASSIGN_VARIABLE
@torch.library.custom_op("tfl::assign_variable", mutates_args=())
def tfl_assign_variable(resource: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    # Assign value to a variable resource - return the value
    return value


@tfl_assign_variable.register_fake
def _(resource, value):
    return value


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
    # Updates a slice of operand with update at start_indices
    output = operand.clone()

    # Build slices for the update region
    slices = []
    for i, start_idx in enumerate(start_indices):
        end_idx = start_idx + update.shape[i]
        slices.append(slice(start_idx, end_idx))

    output[tuple(slices)] = update
    return output


@tfl_dynamic_update_slice.register_fake
def _(operand, update, start_indices):
    return operand.clone()


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
    # Computes the product along segments (unsorted)
    output_shape = [num_segments] + list(data.shape[1:])
    output = torch.ones(output_shape, dtype=data.dtype)

    for i in range(len(segment_ids)):
        seg_id = int(segment_ids[i])
        if seg_id >= 0:  # Negative segment IDs are ignored
            output[seg_id] *= data[i]

    return output


@tfl_unsorted_segment_prod.register_fake
def _(data, segment_ids, num_segments):
    output_shape = [num_segments] + list(data.shape[1:])
    return torch.ones(output_shape, dtype=data.dtype)


# 154: UNSORTED_SEGMENT_MAX
@torch.library.custom_op("tfl::unsorted_segment_max", mutates_args=())
def tfl_unsorted_segment_max(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    # Computes the max along segments (unsorted)
    output_shape = [num_segments] + list(data.shape[1:])
    output = torch.full(output_shape, float('-inf'), dtype=data.dtype)

    for i in range(len(segment_ids)):
        seg_id = int(segment_ids[i])
        if seg_id >= 0:
            output[seg_id] = torch.maximum(output[seg_id], data[i])

    return output


@tfl_unsorted_segment_max.register_fake
def _(data, segment_ids, num_segments):
    output_shape = [num_segments] + list(data.shape[1:])
    return torch.full(output_shape, float('-inf'), dtype=data.dtype)


# 155: UNSORTED_SEGMENT_SUM
@torch.library.custom_op("tfl::unsorted_segment_sum", mutates_args=())
def tfl_unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    # Computes the sum along segments (unsorted)
    output_shape = [num_segments] + list(data.shape[1:])
    output = torch.zeros(output_shape, dtype=data.dtype)

    for i in range(len(segment_ids)):
        seg_id = int(segment_ids[i])
        if seg_id >= 0:
            output[seg_id] += data[i]

    return output


@tfl_unsorted_segment_sum.register_fake
def _(data, segment_ids, num_segments):
    output_shape = [num_segments] + list(data.shape[1:])
    return torch.zeros(output_shape, dtype=data.dtype)


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
    # Computes the min along segments (unsorted)
    output_shape = [num_segments] + list(data.shape[1:])
    output = torch.full(output_shape, float('inf'), dtype=data.dtype)

    for i in range(len(segment_ids)):
        seg_id = int(segment_ids[i])
        if seg_id >= 0:
            output[seg_id] = torch.minimum(output[seg_id], data[i])

    return output


@tfl_unsorted_segment_min.register_fake
def _(data, segment_ids, num_segments):
    output_shape = [num_segments] + list(data.shape[1:])
    return torch.full(output_shape, float('inf'), dtype=data.dtype)


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
@torch.library.custom_op("tfl::dilate", mutates_args=())
def tfl_dilate(x: torch.Tensor, dilations: list[int], padding_value: float) -> torch.Tensor:
    # Dilates a tensor by inserting padding_value between elements
    # dilations: list of dilation factors for each dimension
    output = x
    for dim, dilation in enumerate(dilations):
        if dilation > 1:
            # Insert (dilation - 1) padding values between each element
            shape = list(output.shape)
            new_size = shape[dim] + (shape[dim] - 1) * (dilation - 1)
            new_shape = shape[:dim] + [new_size] + shape[dim+1:]
            dilated = torch.full(new_shape, padding_value, dtype=x.dtype, device=x.device)

            # Copy original values at dilated positions
            slices = [slice(None)] * len(new_shape)
            slices[dim] = slice(None, None, dilation)
            dilated[tuple(slices)] = output
            output = dilated

    return output


@tfl_dilate.register_fake
def _(x, dilations, padding_value):
    shape = list(x.shape)
    for dim, dilation in enumerate(dilations):
        if dilation > 1:
            shape[dim] = shape[dim] + (shape[dim] - 1) * (dilation - 1)
    return torch.empty(shape, dtype=x.dtype)
