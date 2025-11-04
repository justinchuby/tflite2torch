from __future__ import annotations

import tflite2torch


# Convert a TFLite model to a PyTorch model
tflite2torch.convert_tflite_to_torch("examples/yamnet.tflite", "examples/pt_model")
