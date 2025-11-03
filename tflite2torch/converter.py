"""
Main converter module for TFLite to PyTorch conversion.

This module provides the high-level API for converting TFLite models
to PyTorch models, utilizing all four stages of the conversion pipeline.
"""

from typing import Optional, Dict, Union
import os
import torch
from torch.fx import GraphModule

from .parser import TFLiteParser
from .operator_converter import OperatorConverter
from .fx_reconstructor import FXReconstructor
from .code_renderer import CodeRenderer


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
        self.code_renderer = CodeRenderer()

    def convert(
        self,
        tflite_model_path: str,
        output_path: Optional[str] = None,
        generate_code: bool = True,
        subgraph_index: int = 0,
    ) -> Union[GraphModule, str]:
        """
        Convert a TFLite model to PyTorch.

        Args:
            tflite_model_path: Path to the TFLite model file (.tflite)
            output_path: Optional path to save generated PyTorch code
            generate_code: Whether to generate Python code (vs returning GraphModule)
            subgraph_index: Index of the subgraph to convert (default: 0)

        Returns:
            GraphModule if generate_code is False, otherwise the generated code as string

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
        graph_module = self.fx_reconstructor.reconstruct(subgraph)
        print("  Graph reconstruction complete")
        
        # Optionally visualize the graph
        graph_viz = self.fx_reconstructor.visualize_graph(graph_module)
        print("\n" + graph_viz)
        
        # Stage 4: Render to code (if requested)
        if generate_code:
            print("\nStage 4: Rendering PyTorch code...")
            code = self.code_renderer.render(graph_module, class_name="ConvertedModel")
            print("  Code generation complete")
            
            if output_path:
                self.code_renderer.save_to_file(code, output_path)
                print(f"  Saved to {output_path}")
            
            return code
        else:
            return graph_module

    def convert_and_save(
        self,
        tflite_model_path: str,
        output_code_path: str,
        subgraph_index: int = 0,
    ) -> str:
        """
        Convert TFLite model and save the generated code to a file.

        Args:
            tflite_model_path: Path to the TFLite model file
            output_code_path: Path where to save the generated PyTorch code
            subgraph_index: Index of the subgraph to convert

        Returns:
            Generated PyTorch code as string
        """
        return self.convert(
            tflite_model_path=tflite_model_path,
            output_path=output_code_path,
            generate_code=True,
            subgraph_index=subgraph_index,
        )

    def convert_to_graph_module(
        self,
        tflite_model_path: str,
        subgraph_index: int = 0,
    ) -> GraphModule:
        """
        Convert TFLite model to PyTorch GraphModule without code generation.

        Args:
            tflite_model_path: Path to the TFLite model file
            subgraph_index: Index of the subgraph to convert

        Returns:
            PyTorch FX GraphModule
        """
        return self.convert(
            tflite_model_path=tflite_model_path,
            generate_code=False,
            subgraph_index=subgraph_index,
        )


def convert_tflite_to_torch(
    tflite_model_path: str,
    output_path: Optional[str] = None,
    generate_code: bool = True,
    subgraph_index: int = 0,
) -> Union[GraphModule, str]:
    """
    Convenience function to convert a TFLite model to PyTorch.

    This is the main entry point for the library. It performs the complete
    four-stage conversion process:
    1. TFLite graph parsing
    2. TFLite to Torch operator conversion
    3. Reconstruction of the TFLite execution graph in Torch FX
    4. Rendering of the Torch FX graph into Torch code

    Args:
        tflite_model_path: Path to the TFLite model file (.tflite)
        output_path: Optional path to save generated PyTorch code
        generate_code: Whether to generate Python code (default: True)
        subgraph_index: Index of the subgraph to convert (default: 0)

    Returns:
        If generate_code is True: Generated PyTorch code as string
        If generate_code is False: PyTorch FX GraphModule

    Example:
        >>> # Convert to code
        >>> code = convert_tflite_to_torch("model.tflite", "model.py")
        >>> 
        >>> # Convert to GraphModule
        >>> graph_module = convert_tflite_to_torch("model.tflite", generate_code=False)
        >>> output = graph_module(input_tensor)
    """
    converter = TFLiteToTorchConverter()
    return converter.convert(
        tflite_model_path=tflite_model_path,
        output_path=output_path,
        generate_code=generate_code,
        subgraph_index=subgraph_index,
    )
