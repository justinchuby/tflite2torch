"""
Main converter module for TFLite to PyTorch conversion.

This module provides the high-level API for converting TFLite models
to PyTorch models, utilizing all four stages of the conversion pipeline.
"""

import os
import torch
from torch.fx import GraphModule

from ._parser import TFLiteParser
from ._operator_converter import OperatorConverter
from ._fx_reconstructor import FXReconstructor


class _TFLiteToTorchConverter:
    """
    Main converter class for TFLite to PyTorch conversion.

    This class orchestrates the four-stage conversion process:
    1. Parse TFLite model
    2. Convert operators
    3. Reconstruct FX graph
    4. Render to code
    """

    def __init__(self):
        # The class is created so that we can observe states for debugging
        self.parser = TFLiteParser()
        self.operator_converter = OperatorConverter()
        self.fx_reconstructor = FXReconstructor(self.operator_converter)

    def convert(self, tflite_model_path: str, subgraph_index: int = 0) -> GraphModule:
        """
        Convert a TFLite model to PyTorch.

        Args:
            tflite_model_path: Path to the TFLite model file (.tflite)
            subgraph_index: Index of the subgraph to convert (default: 0)

        Returns:
            GraphModule.

        Raises:
            FileNotFoundError: If the TFLite model file doesn't exist
            ValueError: If the model is invalid or conversion fails
        """
        # Validate input
        if not os.path.exists(tflite_model_path):
            raise FileNotFoundError(f"TFLite model not found: {tflite_model_path}")

        # Stage 1: Parse TFLite model
        print(f"Stage 1: Parsing TFLite model from {tflite_model_path}...")
        subgraphs = self.parser.parse(tflite_model_path)

        if not subgraphs:
            raise ValueError("No subgraphs found in TFLite model")

        if subgraph_index >= len(subgraphs):
            raise ValueError(
                f"Subgraph index {subgraph_index} out of range. "
                f"Model has {len(subgraphs)} subgraph(s)."
            )

        subgraph = subgraphs[subgraph_index]
        print(f"  Found {len(subgraphs)} subgraph(s)")
        print(f"  Converting subgraph {subgraph_index}: {subgraph.name}")
        print(f"  - {len(subgraph.tensors)} tensors")
        print(f"  - {len(subgraph.operators)} operators")

        # Stage 2 & 3: Convert operators and reconstruct FX graph
        print("\nStage 2-3: Converting operators and reconstructing FX graph...")
        # Get weights from parser
        weights_dict = self.parser.get_weights(subgraph_index)
        # Convert numpy arrays to torch tensors
        weights_torch = {idx: torch.from_numpy(weight) for idx, weight in weights_dict.items()}
        graph_module = self.fx_reconstructor.reconstruct(subgraph, weights=weights_torch)
        print("  Graph reconstruction complete")

        return graph_module


def convert_tflite_to_torch(
    tflite_model_path: str, output_path: str, subgraph_index: int = 0
) -> None:
    """
    Convert a TFLite model to PyTorch code.

    This function performs the complete four-stage conversion process:
    1. TFLite graph parsing
    2. TFLite to Torch operator conversion
    3. Reconstruction of the TFLite execution graph in Torch FX
    4. Rendering of the Torch FX graph into Torch code

    Args:
        tflite_model_path: Path to the TFLite model file (.tflite)
        output_path: Path to save generated PyTorch code
        subgraph_index: Index of the subgraph to convert (default: 0)

    Returns:
        Generated PyTorch code as string

    Example:
        >>> # Convert and save to folder
        >>> convert_tflite_to_torch("model.tflite", "output_folder")
    """
    converter = _TFLiteToTorchConverter()
    graph_module = converter.convert(
        tflite_model_path=tflite_model_path,
        subgraph_index=subgraph_index,
    )
    graph_module.to_folder(output_path)
    print(f"PyTorch model saved to {output_path}")


def convert_tflite_to_graph_module(tflite_model_path: str, subgraph_index: int = 0) -> GraphModule:
    """
    Convert a TFLite model to PyTorch FX GraphModule.

    This function performs three stages of the conversion process:
    1. TFLite graph parsing
    2. TFLite to Torch operator conversion
    3. Reconstruction of the TFLite execution graph in Torch FX

    Args:
        tflite_model_path: Path to the TFLite model file (.tflite)
        subgraph_index: Index of the subgraph to convert (default: 0)

    Returns:
        PyTorch FX GraphModule that can be executed directly

    Example:
        >>> # Convert to GraphModule
        >>> graph_module = convert_tflite_to_graph_module("model.tflite")
        >>> output = graph_module(input_tensor)
    """
    converter = _TFLiteToTorchConverter()
    return converter.convert(tflite_model_path=tflite_model_path, subgraph_index=subgraph_index)


def convert_tflite_to_exported_program(tflite_model_path: str, subgraph_index: int = 0):
    """
    Convert a TFLite model to torch.export.ExportedProgram.

    This function performs three stages of the conversion process and then
    exports the result as an ExportedProgram:
    1. TFLite graph parsing
    2. TFLite to Torch operator conversion
    3. Reconstruction of the TFLite execution graph in Torch FX
    4. Export to ExportedProgram

    Args:
        tflite_model_path: Path to the TFLite model file (.tflite)
        subgraph_index: Index of the subgraph to convert (default: 0)

    Returns:
        torch.export.ExportedProgram if torch.export is available, otherwise None

    Example:
        >>> ep = convert_tflite_to_exported_program("model.tflite")

    Note:
        Requires PyTorch 2.7+ with torch.export support.
    """
    converter = _TFLiteToTorchConverter()
    graph_module = converter.convert(
        tflite_model_path=tflite_model_path, subgraph_index=subgraph_index
    )
