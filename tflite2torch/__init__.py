"""
tflite2torch: Convert TensorFlow Lite models to PyTorch

This library provides functionality to convert TFLite models to PyTorch models
through a four-stage process:
1. TFLite graph parsing
2. TFLite to Torch operator conversion
3. Reconstruction of the TFLite execution graph in Torch FX
4. Rendering of the Torch FX graph into Torch code
"""

__version__ = "0.1.0"

from .converter import (
    convert_tflite_to_torch,
    convert_tflite_to_graph_module,
    convert_tflite_to_exported_program,
    TFLiteToTorchConverter,
)

__all__ = [
    "convert_tflite_to_torch",
    "convert_tflite_to_graph_module",
    "convert_tflite_to_exported_program",
    "TFLiteToTorchConverter",
]
