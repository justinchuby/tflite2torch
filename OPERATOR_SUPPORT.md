# TFLite Operator Support

This document lists all TensorFlow Lite operators supported by tflite2torch, based on the official TensorFlow Lite MLIR specification: https://www.tensorflow.org/mlir/tfl_ops

## Overview

**Total Operators Supported: 116+**

All operators are sourced from the official `tfl_ops.td` file in the TensorFlow repository and mapped to their PyTorch equivalents.

## Operator Categories

### Arithmetic & Math Operations (32 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| ABS | torch.abs | ✅ Supported |
| ADD | torch.add | ✅ Supported |
| ADD_N | torch.sum | ✅ Supported |
| CEIL | torch.ceil | ✅ Supported |
| COS | torch.cos | ✅ Supported |
| DIV | torch.div | ✅ Supported |
| EXP | torch.exp | ✅ Supported |
| FLOOR | torch.floor | ✅ Supported |
| FLOOR_DIV | torch.floor_divide | ✅ Supported |
| FLOOR_MOD | torch.fmod | ✅ Supported |
| LOG | torch.log | ✅ Supported |
| MAXIMUM | torch.maximum | ✅ Supported |
| MINIMUM | torch.minimum | ✅ Supported |
| MUL | torch.mul | ✅ Supported |
| NEG | torch.neg | ✅ Supported |
| POW | torch.pow | ✅ Supported |
| RSQRT | torch.rsqrt | ✅ Supported |
| SIN | torch.sin | ✅ Supported |
| SQRT | torch.sqrt | ✅ Supported |
| SQUARE | torch.square | ✅ Supported |
| SQUARED_DIFFERENCE | Custom | ✅ Supported |
| SUB | torch.sub | ✅ Supported |

### Convolution & Pooling Operations (6 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| AVERAGE_POOL_2D | nn.AvgPool2d | ✅ Supported |
| CONV_2D | nn.Conv2d | ✅ Supported |
| CONV_3D | nn.Conv3d | ✅ Supported |
| DEPTHWISE_CONV_2D | nn.Conv2d (groups) | ✅ Supported |
| MAX_POOL_2D | nn.MaxPool2d | ✅ Supported |
| TRANSPOSE_CONV | nn.ConvTranspose2d | ✅ Supported |

### Fully Connected Operations (2 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| BATCH_MATMUL | torch.bmm | ✅ Supported |
| FULLY_CONNECTED | nn.Linear | ✅ Supported |

### Activation Functions (11 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| ELU | nn.ELU | ✅ Supported |
| GELU | nn.GELU | ✅ Supported |
| HARD_SWISH | nn.Hardswish | ✅ Supported |
| LEAKY_RELU | nn.LeakyReLU | ✅ Supported |
| LOGISTIC | nn.Sigmoid | ✅ Supported |
| LOG_SOFTMAX | nn.LogSoftmax | ✅ Supported |
| PRELU | nn.PReLU | ✅ Supported |
| RELU | nn.ReLU | ✅ Supported |
| RELU6 | nn.ReLU6 | ✅ Supported |
| SOFTMAX | nn.Softmax | ✅ Supported |
| TANH | nn.Tanh | ✅ Supported |

### Normalization Operations (2 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| L2_NORMALIZATION | F.normalize | ✅ Supported |
| LOCAL_RESPONSE_NORMALIZATION | nn.LocalResponseNorm | ✅ Supported |

### Reduction Operations (6 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| MEAN | torch.mean | ✅ Supported |
| REDUCE_ANY | torch.any | ✅ Supported |
| REDUCE_MAX | torch.max | ✅ Supported |
| REDUCE_MIN | torch.min | ✅ Supported |
| REDUCE_PROD | torch.prod | ✅ Supported |
| SUM | torch.sum | ✅ Supported |

### Shape & Tensor Manipulation (36 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| BATCH_TO_SPACE_ND | Custom | ✅ Supported |
| BROADCAST_ARGS | torch.broadcast_shapes | ✅ Supported |
| BROADCAST_TO | torch.broadcast_to | ✅ Supported |
| CONCATENATION | torch.cat | ✅ Supported |
| DEPTH_TO_SPACE | nn.PixelShuffle | ✅ Supported |
| EXPAND_DIMS | torch.unsqueeze | ✅ Supported |
| FILL | torch.full | ✅ Supported |
| GATHER | torch.gather | ✅ Supported |
| GATHER_ND | Custom | ✅ Supported |
| MIRROR_PAD | F.pad (reflect) | ✅ Supported |
| PACK | torch.stack | ✅ Supported |
| PAD | F.pad | ✅ Supported |
| PADV2 | F.pad | ✅ Supported |
| RANGE | torch.arange | ✅ Supported |
| RESHAPE | torch.reshape | ✅ Supported |
| RESIZE_BILINEAR | F.interpolate | ✅ Supported |
| RESIZE_NEAREST_NEIGHBOR | F.interpolate | ✅ Supported |
| REVERSE_SEQUENCE | Custom | ✅ Supported |
| REVERSE_V2 | torch.flip | ✅ Supported |
| SCATTER_ND | torch.scatter | ✅ Supported |
| SHAPE | Custom | ✅ Supported |
| SLICE | torch.slice | ✅ Supported |
| SPACE_TO_BATCH_ND | Custom | ✅ Supported |
| SPACE_TO_DEPTH | nn.PixelUnshuffle | ✅ Supported |
| SPARSE_TO_DENSE | Custom | ✅ Supported |
| SPLIT | torch.split | ✅ Supported |
| SPLIT_V | torch.split | ✅ Supported |
| SQUEEZE | torch.squeeze | ✅ Supported |
| STRIDED_SLICE | Custom | ✅ Supported |
| TILE | torch.tile | ✅ Supported |
| TOPK_V2 | torch.topk | ✅ Supported |
| TRANSPOSE | torch.permute | ✅ Supported |
| UNIQUE | torch.unique | ✅ Supported |
| UNPACK | torch.unbind | ✅ Supported |
| WHERE | torch.where | ✅ Supported |
| ZEROS_LIKE | torch.zeros_like | ✅ Supported |

### Comparison Operations (6 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| EQUAL | torch.eq | ✅ Supported |
| GREATER | torch.gt | ✅ Supported |
| GREATER_EQUAL | torch.ge | ✅ Supported |
| LESS | torch.lt | ✅ Supported |
| LESS_EQUAL | torch.le | ✅ Supported |
| NOT_EQUAL | torch.ne | ✅ Supported |

### Logical Operations (3 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| LOGICAL_AND | torch.logical_and | ✅ Supported |
| LOGICAL_NOT | torch.logical_not | ✅ Supported |
| LOGICAL_OR | torch.logical_or | ✅ Supported |

### Selection Operations (5 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| ARG_MAX | torch.argmax | ✅ Supported |
| ARG_MIN | torch.argmin | ✅ Supported |
| ONE_HOT | F.one_hot | ✅ Supported |
| SELECT | torch.where | ✅ Supported |
| SELECT_V2 | torch.where | ✅ Supported |

### Recurrent Neural Network Operations (6 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| BIDIRECTIONAL_SEQUENCE_LSTM | nn.LSTM (bidirectional) | ✅ Supported |
| BIDIRECTIONAL_SEQUENCE_RNN | nn.RNN (bidirectional) | ✅ Supported |
| LSTM | nn.LSTM | ✅ Supported |
| RNN | nn.RNN | ✅ Supported |
| UNIDIRECTIONAL_SEQUENCE_LSTM | nn.LSTM | ✅ Supported |
| UNIDIRECTIONAL_SEQUENCE_RNN | nn.RNN | ✅ Supported |

### Quantization Operations (3 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| DEQUANTIZE | Custom | ✅ Supported |
| FAKE_QUANT | torch.fake_quantize_per_tensor_affine | ✅ Supported |
| QUANTIZE | torch.quantize_per_tensor | ✅ Supported |

### Type Conversion (1 operator)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| CAST | Custom | ✅ Supported |

### Embedding & Lookup (2 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| EMBEDDING_LOOKUP | nn.Embedding | ✅ Supported |
| HASHTABLE_LOOKUP | Custom | ✅ Supported |

### Advanced Operations (5 operators)

| TFLite Operator | PyTorch Equivalent | Status |
|-----------------|-------------------|---------|
| CUMSUM | torch.cumsum | ✅ Supported |
| CUSTOM | Custom | ✅ Supported |
| MATRIX_DIAG | torch.diag | ✅ Supported |
| MATRIX_SET_DIAG | Custom | ✅ Supported |
| SEGMENT_SUM | Custom | ✅ Supported |

## Implementation Notes

### Direct Mappings
Most operators have direct PyTorch equivalents and can be converted automatically with appropriate parameter mappings.

### Custom Implementations
Some operators marked as "Custom" require custom implementation logic:
- Operations with no direct PyTorch equivalent (e.g., BATCH_TO_SPACE_ND)
- Operations requiring special handling (e.g., STRIDED_SLICE)
- Operations specific to TFLite (e.g., HASHTABLE_LOOKUP)

These operators return a dictionary with `"custom": True` flag, indicating that custom implementation is needed in the graph reconstruction phase.

### Fused Activations
Many TFLite operators support fused activation functions. The converter automatically handles:
- NONE
- RELU
- RELU6
- TANH

### Quantization
Quantization operators are mapped to PyTorch's quantization APIs. Note that full quantization support requires:
- Proper scale and zero-point parameters
- Appropriate tensor data types
- Post-conversion quantization calibration

## Usage

```python
from tflite2torch import OperatorConverter

converter = OperatorConverter()

# Convert a specific operator
result = converter.convert(
    "CONV_2D",
    inputs=[0, 1, 2],
    options={"stride_h": 1, "stride_w": 1, "padding": "SAME"}
)

# Check total supported operators
print(f"Total operators: {len(converter.converters)}")
```

## References

- [TensorFlow Lite MLIR Dialect](https://www.tensorflow.org/mlir/tfl_ops)
- [TFLite Ops Source (tfl_ops.td)](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/ir/tfl_ops.td)
- [TFLite Operator Compatibility](https://ai.google.dev/edge/litert/models/ops_compatibility)

## Contributing

To add support for additional operators:

1. Add the operator to the `_register_converters()` method
2. Implement the conversion method `_convert_<operator_name>()`
3. Return a dictionary with:
   - `module`: PyTorch module/function
   - `params`: Parameters for the module
   - `custom`: (optional) True if custom implementation needed
4. Add tests in `tests/test_operator_converter.py`
5. Update this documentation

## Version History

- **v0.1.0**: Initial release with 30 operators
- **v0.1.1**: Expanded to 116+ operators based on MLIR tfl_ops specification
