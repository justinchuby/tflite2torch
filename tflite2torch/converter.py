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


class TFLiteToTorchConverter:
    """
    Main converter class for TFLite to PyTorch conversion.

    This class orchestrates the four-stage conversion process:
    1. Parse TFLite model
    2. Convert operators
    3. Reconstruct FX graph
    4. Render to code
    """

    def __init__(self):
        self.parser = TFLiteParser()
        self.operator_converter = OperatorConverter()
        self.fx_reconstructor = FXReconstructor(self.operator_converter)

    def convert(
        self,
        tflite_model_path: str,
        subgraph_index: int = 0,
    ) -> GraphModule:
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

    def convert_and_save(
        self,
        tflite_model_path: str,
        output_path: str,
        subgraph_index: int = 0,
    ) -> None:
        """
        Convert TFLite model and save the generated code to a file.

        Args:
            tflite_model_path: Path to the TFLite model file
            output_path: Path where to save the generated PyTorch code
            subgraph_index: Index of the subgraph to convert

        Returns:
            Generated PyTorch code as string
        """
        graph_module = self.convert(
            tflite_model_path=tflite_model_path, subgraph_index=subgraph_index
        )
        graph_module.to_folder(output_path)


def convert_tflite_to_torch(
    tflite_model_path: str,
    output_path: str,
    subgraph_index: int = 0,
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
        >>> # Convert to code
        >>> code = convert_tflite_to_torch("model.tflite")
        >>>
        >>> # Convert and save to file
        >>> code = convert_tflite_to_torch("model.tflite", "model.py")
    """
    converter = TFLiteToTorchConverter()
    converter.convert_and_save(
        tflite_model_path=tflite_model_path,
        output_path=output_path,
        subgraph_index=subgraph_index,
    )


def convert_tflite_to_graph_module(
    tflite_model_path: str,
    subgraph_index: int = 0,
) -> GraphModule:
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
    converter = TFLiteToTorchConverter()
    return converter.convert(tflite_model_path=tflite_model_path, subgraph_index=subgraph_index)


def convert_tflite_to_exported_program(
    tflite_model_path: str,
    subgraph_index: int = 0,
):
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
        example_inputs: Tuple of example input tensors for export.
                       If not provided, creates dummy tensors (not recommended for production)
        subgraph_index: Index of the subgraph to convert (default: 0)

    Returns:
        torch.export.ExportedProgram if torch.export is available, otherwise None

    Example:
        >>> # Convert to ExportedProgram with example inputs
        >>> example_inputs = (torch.randn(1, 3, 224, 224),)
        >>> exported = convert_tflite_to_exported_program("model.tflite", example_inputs)
        >>>
        >>> # Let it create dummy inputs (not recommended for production)
        >>> exported = convert_tflite_to_exported_program("model.tflite")

    Note:
        Requires PyTorch 2.7+ with torch.export support.
    """
    converter = TFLiteToTorchConverter()
    graph_module = converter.convert(
        tflite_model_path=tflite_model_path, subgraph_index=subgraph_index
    )
