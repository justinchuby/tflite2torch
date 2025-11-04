from __future__ import annotations

import tflite2torch


# Convert a TFLite model to a PyTorch model
graph_module = tflite2torch.convert_tflite_to_graph_module("examples/yamnet.tflite")
print(graph_module.graph)

# Save the converted PyTorch model to a folder
tflite2torch.convert_tflite_to_torch("examples/yamnet.tflite", "examples/pt_model")
