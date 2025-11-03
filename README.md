# tflite2torch

A Python library for converting TensorFlow Lite (TFLite) models to PyTorch models.

## Overview

`tflite2torch` provides a complete pipeline for converting TFLite models to PyTorch through a four-stage architecture:

1. **TFLite Graph Parsing** - Parse TFLite model files and extract the computational graph structure
2. **TFLite to Torch Operator Conversion** - Map TFLite operators to their PyTorch equivalents
3. **Torch FX Graph Reconstruction** - Reconstruct the execution graph using PyTorch FX (and optionally export as `torch.export.ExportedProgram`)
4. **Code Rendering** - Generate readable PyTorch code from the FX graph

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
from tflite2torch import convert_tflite_to_torch

# Convert TFLite model to PyTorch code
code = convert_tflite_to_torch("model.tflite", output_path="model.py")

# Or convert to a PyTorch FX GraphModule
graph_module = convert_tflite_to_torch("model.tflite", generate_code=False)
```

### Using the Converter Class

```python
from tflite2torch import TFLiteToTorchConverter

converter = TFLiteToTorchConverter()

# Generate code
code = converter.convert_and_save("model.tflite", "output_model.py")

# Or get a GraphModule for direct execution
graph_module = converter.convert_to_graph_module("model.tflite")
output = graph_module(input_tensor)
```

## Architecture

### Stage 1: TFLite Graph Parsing

The `TFLiteParser` class handles parsing of TFLite FlatBuffer format:

```python
from tflite2torch import TFLiteParser

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
from tflite2torch import OperatorConverter

converter = OperatorConverter()
conv_info = converter.convert(
    "CONV_2D",
    inputs=[0, 1, 2],
    options={"stride_h": 1, "stride_w": 1, "padding": "SAME"}
)
```

Supported operators include:
- Convolution: CONV_2D, DEPTHWISE_CONV_2D
- Fully Connected: FULLY_CONNECTED
- Activation: RELU, RELU6, TANH, SIGMOID, SOFTMAX
- Pooling: MAX_POOL_2D, AVERAGE_POOL_2D
- Element-wise: ADD, MUL, SUB, DIV
- Shape manipulation: RESHAPE, TRANSPOSE, CONCATENATION, SQUEEZE, EXPAND_DIMS
- And many more...

### Stage 3: FX Graph Reconstruction

The `FXReconstructor` class rebuilds the computation graph using PyTorch FX:

```python
from tflite2torch import FXReconstructor

reconstructor = FXReconstructor()
graph_module = reconstructor.reconstruct(subgraph, weights)

# Visualize the graph
print(reconstructor.visualize_graph(graph_module))
```

### Stage 4: Code Rendering

The `CodeRenderer` class generates readable PyTorch code:

```python
from tflite2torch import CodeRenderer

renderer = CodeRenderer()
code = renderer.render(graph_module, class_name="MyModel")
renderer.save_to_file(code, "my_model.py")
```

## Examples

See the `examples/` directory for complete examples:

```bash
python examples/basic_conversion.py
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

### Main Conversion Function

```python
convert_tflite_to_torch(
    tflite_model_path: str,
    output_path: Optional[str] = None,
    generate_code: bool = True,
    subgraph_index: int = 0
) -> Union[GraphModule, str]
```

**Parameters:**
- `tflite_model_path`: Path to the TFLite model file (.tflite)
- `output_path`: Optional path to save generated PyTorch code
- `generate_code`: Whether to generate Python code (default: True)
- `subgraph_index`: Index of the subgraph to convert (default: 0)

**Returns:**
- If `generate_code=True`: Generated PyTorch code as string
- If `generate_code=False`: PyTorch FX GraphModule

### Classes

- `TFLiteParser`: Parse TFLite model files
- `OperatorConverter`: Convert TFLite operators to PyTorch
- `FXReconstructor`: Reconstruct computation graph in PyTorch FX
- `CodeRenderer`: Generate PyTorch code from FX graph
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