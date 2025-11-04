"""
Main converter module for TFLite to PyTorch conversion.

This module provides the high-level API for converting TFLite models
to PyTorch models, utilizing all four stages of the conversion pipeline.
"""

from __future__ import annotations

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
    converter = TFLiteToTorchConverter()
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
    converter = TFLiteToTorchConverter()
    return converter.convert(tflite_model_path=tflite_model_path, subgraph_index=subgraph_index)


def convert_tflite_to_exported_program(tflite_model_path: str, subgraph_index: int = 0):
    """
    Convert a TFLite model to torch.export.ExportedProgram.

    This function performs three stages of the conversion process and then
    exports the result as an ExportedProgram:
    1. TFLite graph parsing
    2. TFLite to Torch operator conversion
    3. Reconstruction of the TFLite execution graph in Torch FX
    4. Export to ExportedProgram with proper input signature

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

    # TFLite dtype to PyTorch dtype mapping
    dtype_str_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "int32": torch.int32,
        "uint8": torch.uint8,
        "int64": torch.int64,
        "bool": torch.bool,
        "int16": torch.int16,
        "complex64": torch.complex64,
        "int8": torch.int8,
        "complex128": torch.complex128,
    }

    converter = TFLiteToTorchConverter()
    parser = TFLiteParser()

    # Get input tensor information from the TFLite model
    input_tensors = parser.parse(tflite_model_path)
    if not input_tensors or subgraph_index >= len(input_tensors):
        raise ValueError(f"Invalid subgraph index {subgraph_index}")

    # Get input tensor specs for the specified subgraph
    input_tensor_infos = parser.get_input_tensors(subgraph_index)

    # Create example inputs based on input tensor specifications
    example_inputs_1 = []
    example_inputs_2 = []

    for i, tensor_info in enumerate(input_tensor_infos):
        # Convert TFLite dtype to PyTorch dtype
        torch_dtype = dtype_str_map[tensor_info.dtype]

        # Handle dynamic shapes
        example_shape_1 = tensor_info.shape.copy()
        example_shape_2 = tensor_info.shape.copy()

        # Replace any -1 dimensions with a concrete size for example input
        for j, dim in enumerate(example_shape_1):
            if dim == -1:
                if j == 0:  # batch dimension
                    example_shape_1[j] = 2
                else:
                    example_shape_1[j] = 224  # Default size 1 for other unknown dimensions

        for j, dim in enumerate(example_shape_2):
            if dim == -1:
                if j == 0:  # batch dimension
                    example_shape_2[j] = 3
                else:
                    example_shape_2[j] = 256  # Default size 2 for other unknown dimensions

        # Create example input tensor
        example_input_1 = torch.randn(example_shape_1, dtype=torch_dtype)
        example_input_2 = torch.randn(example_shape_2, dtype=torch_dtype)
        example_inputs_1.append(example_input_1)
        example_inputs_2.append(example_input_2)

    dynamic_shapes = torch.export.AdditionalInputs()
    dynamic_shapes.add(*example_inputs_1)
    dynamic_shapes.add(*example_inputs_2)

    # Convert TFLite model to GraphModule
    graph_module = converter.convert(
        tflite_model_path=tflite_model_path, subgraph_index=subgraph_index
    )

    return torch.export.export(graph_module, tuple(example_inputs_1), dynamic_shapes=dynamic_shapes)
