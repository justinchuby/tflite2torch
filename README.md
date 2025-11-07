# tflite2torch

A Python library for converting TensorFlow Lite (TFLite) models to PyTorch models.

## Overview

`tflite2torch` provides a complete pipeline for converting TFLite models to PyTorch through a three-stage architecture:

1. **TFLite Graph Parsing** - Parse TFLite model files and extract the computational graph structure
2. **TFLite to Torch Operator Conversion** - Map TFLite operators to their PyTorch equivalents
3. **Torch FX Graph Reconstruction** - Reconstruct the execution graph using PyTorch FX (and optionally export as `torch.export.ExportedProgram`)

## Installation

### From source:

```bash
git clone https://github.com/justinchuby/tflite2torch.git
cd tflite2torch
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.20.0

## Quick Start

### Basic Conversion

```python
import tflite2torch

# Convert TFLite model to a PyTorch FX GraphModule
graph_module = tflite2torch.convert_tflite_to_graph_module("model.tflite")
print(graph_module.graph)

# Save the converted PyTorch model to a folder
tflite2torch.convert_tflite_to_torch("model.tflite", "output_folder")
```

### Converting to ExportedProgram

```python
import tflite2torch

# Convert TFLite model to torch.export.ExportedProgram
exported_program = tflite2torch.convert_tflite_to_exported_program("model.tflite")
```

## Architecture

### Stage 1: TFLite Graph Parsing

The `TFLiteParser` class handles parsing of TFLite FlatBuffer format:

```python
from tflite2torch._parser import TFLiteParser

parser = TFLiteParser()
subgraphs = parser.parse("model.tflite")

# Access parsed information
for subgraph in subgraphs:
    print(f"Subgraph: {subgraph.name}")
    print(f"  Tensors: {len(subgraph.tensors)}")
    print(f"  Operators: {len(subgraph.operators)}")
```

### Stage 2: Operator Conversion

The `OperatorConverter` class maps TFLite operators to PyTorch:

```python
from tflite2torch._operator_converter import OperatorConverter

converter = OperatorConverter()
# Internal usage - typically called by FXReconstructor
# Supports 116+ TFLite operators
```

**Comprehensive Operator Support (116+ operators)**:

Based on the official TensorFlow Lite MLIR specification (https://www.tensorflow.org/mlir/tfl_ops), we support:

- **Arithmetic & Math** (32 ops): ABS, ADD, ADD_N, CEIL, COS, DIV, EXP, FLOOR, FLOOR_DIV, FLOOR_MOD, LOG, MAXIMUM, MINIMUM, MUL, NEG, POW, RSQRT, SIN, SQRT, SQUARE, SQUARED_DIFFERENCE, SUB, and more
- **Convolution & Pooling** (6 ops): AVERAGE_POOL_2D, CONV_2D, CONV_3D, DEPTHWISE_CONV_2D, MAX_POOL_2D, TRANSPOSE_CONV
- **Fully Connected** (2 ops): FULLY_CONNECTED, BATCH_MATMUL
- **Activation Functions** (11 ops): ELU, GELU, HARD_SWISH, LEAKY_RELU, LOGISTIC, LOG_SOFTMAX, PRELU, RELU, RELU6, SOFTMAX, TANH
- **Normalization** (2 ops): L2_NORMALIZATION, LOCAL_RESPONSE_NORMALIZATION
- **Reduction Operations** (6 ops): MEAN, REDUCE_MAX, REDUCE_MIN, REDUCE_PROD, REDUCE_ANY, SUM
- **Shape & Tensor Manipulation** (36 ops): BATCH_TO_SPACE_ND, BROADCAST_TO, CONCATENATION, EXPAND_DIMS, GATHER, GATHER_ND, PAD, RESHAPE, SLICE, SPLIT, SQUEEZE, TILE, TRANSPOSE, and more
- **Comparison Operations** (6 ops): EQUAL, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL, NOT_EQUAL
- **Logical Operations** (3 ops): LOGICAL_AND, LOGICAL_NOT, LOGICAL_OR
- **Selection Operations** (5 ops): ARG_MAX, ARG_MIN, ONE_HOT, SELECT, SELECT_V2
- **Recurrent Neural Networks** (6 ops): LSTM, BIDIRECTIONAL_SEQUENCE_LSTM, UNIDIRECTIONAL_SEQUENCE_LSTM, RNN, BIDIRECTIONAL_SEQUENCE_RNN, UNIDIRECTIONAL_SEQUENCE_RNN
- **Quantization** (3 ops): QUANTIZE, DEQUANTIZE, FAKE_QUANT
- **Type Conversion** (1 op): CAST
- **Embedding & Lookup** (2 ops): EMBEDDING_LOOKUP, HASHTABLE_LOOKUP
- **Advanced Operations** (5 ops): CUSTOM, CUMSUM, MATRIX_DIAG, MATRIX_SET_DIAG, SEGMENT_SUM

### Stage 3: FX Graph Reconstruction

The `FXReconstructor` class rebuilds the computation graph using PyTorch FX:

```python
from tflite2torch._fx_reconstructor import FXReconstructor
from tflite2torch._parser import TFLiteParser
from tflite2torch._operator_converter import OperatorConverter

parser = TFLiteParser()
subgraphs = parser.parse("model.tflite")
weights = parser.get_weights(0)

operator_converter = OperatorConverter()
reconstructor = FXReconstructor(operator_converter)
graph_module = reconstructor.reconstruct(subgraphs[0], weights)

# Visualize the graph
print(graph_module.graph)
```

## Examples

See the `examples/` directory for complete examples:

- `examples/yamnet.py` - Convert YAMNet audio classification model
- `examples/birdnet.py` - Convert BirdNET bird sound classification model

```bash
python examples/yamnet.py
```

## Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=tflite2torch --cov-report=html
```

## API Reference

### Main Conversion Functions

#### `convert_tflite_to_torch(tflite_model_path, output_path, subgraph_index=0)`

Convert a TFLite model to PyTorch and save to a folder.

```python
convert_tflite_to_torch(
    tflite_model_path: str,
    output_path: str,
    subgraph_index: int = 0
) -> None
```

**Parameters:**
- `tflite_model_path`: Path to the TFLite model file (.tflite)
- `output_path`: Path to folder where the PyTorch model will be saved
- `subgraph_index`: Index of the subgraph to convert (default: 0)

**Returns:** None (saves model to the specified folder)

**Example:**
```python
import tflite2torch
tflite2torch.convert_tflite_to_torch("model.tflite", "output_folder")
```

#### `convert_tflite_to_graph_module(tflite_model_path, subgraph_index=0)`

Convert a TFLite model to a PyTorch FX GraphModule.

```python
convert_tflite_to_graph_module(
    tflite_model_path: str,
    subgraph_index: int = 0
) -> GraphModule
```

**Parameters:**
- `tflite_model_path`: Path to the TFLite model file (.tflite)
- `subgraph_index`: Index of the subgraph to convert (default: 0)

**Returns:** PyTorch FX GraphModule that can be executed directly

**Example:**
```python
import tflite2torch
graph_module = tflite2torch.convert_tflite_to_graph_module("model.tflite")
output = graph_module(input_tensor)
```

#### `convert_tflite_to_exported_program(tflite_model_path, subgraph_index=0)`

Convert a TFLite model to torch.export.ExportedProgram.

```python
convert_tflite_to_exported_program(
    tflite_model_path: str,
    subgraph_index: int = 0
) -> ExportedProgram
```

**Parameters:**
- `tflite_model_path`: Path to the TFLite model file (.tflite)
- `subgraph_index`: Index of the subgraph to convert (default: 0)

**Returns:** torch.export.ExportedProgram (requires PyTorch 2.7+)

**Example:**
```python
import tflite2torch
exported_program = tflite2torch.convert_tflite_to_exported_program("model.tflite")
```

### Internal Classes

These classes are used internally by the conversion pipeline:

- `TFLiteParser`: Parse TFLite model files
- `OperatorConverter`: Convert TFLite operators to PyTorch
- `FXReconstructor`: Reconstruct computation graph in PyTorch FX
- `TFLiteToTorchConverter`: Main converter orchestrating all stages

## Limitations

- This is a demonstration implementation. For production use, consider using the official TFLite schema for parsing
- Not all TFLite operators are currently supported
- Quantized models may require additional handling
- Custom operators need custom conversion implementations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This library demonstrates the architecture for TFLite to PyTorch conversion using modern PyTorch features like FX and torch.export.