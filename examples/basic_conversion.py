"""
Basic example of converting a TFLite model to PyTorch.

This example demonstrates the three main conversion APIs:
1. convert_tflite_to_torch - Generates PyTorch code
2. convert_tflite_to_graph_module - Creates a GraphModule for execution
3. convert_tflite_to_exported_program - Creates an ExportedProgram
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tflite2torch import (
    convert_tflite_to_torch,
    convert_tflite_to_graph_module,
    convert_tflite_to_exported_program,
)


def main():
    """Main example function."""
    print("=" * 70)
    print("TFLite to PyTorch Conversion Example")
    print("=" * 70)
    print()
    
    # For this example, we'll use a mock TFLite file
    # In real usage, you would provide a path to an actual .tflite file
    mock_tflite_path = "/tmp/example_model.tflite"
    
    # Create a minimal mock file (in real usage, this would be a real TFLite model)
    os.makedirs(os.path.dirname(mock_tflite_path), exist_ok=True)
    with open(mock_tflite_path, "wb") as f:
        # Write some dummy bytes to simulate a TFLite file
        f.write(b"TFL3" + b"\x00" * 100)
    
    print(f"Converting TFLite model: {mock_tflite_path}")
    print()
    
    # Method 1: Generate PyTorch code
    print("Method 1: Generate PyTorch code")
    print("-" * 70)
    code = convert_tflite_to_torch(
        tflite_model_path=mock_tflite_path,
        output_path="/tmp/converted_model.py"
    )
    print("\nGenerated code:")
    print(code)
    print()
    
    # Method 2: Convert to GraphModule for direct execution
    print("\n" + "=" * 70)
    print("Method 2: Convert to GraphModule")
    print("-" * 70)
    graph_module = convert_tflite_to_graph_module(
        tflite_model_path=mock_tflite_path
    )
    
    print(f"\nGraphModule type: {type(graph_module)}")
    print(f"GraphModule: {graph_module}")
    
    # Test the model with sample input
    print("\nTesting the converted model...")
    try:
        sample_input = torch.randn(1, 224, 224, 3)
        output = graph_module(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        print("✓ Model execution successful!")
    except Exception as e:
        print(f"Note: Model execution test: {e}")
    
    # Method 3: Convert to ExportedProgram (requires PyTorch 2.0+)
    print("\n" + "=" * 70)
    print("Method 3: Convert to ExportedProgram")
    print("-" * 70)
    try:
        # Provide example inputs for export
        example_inputs = (torch.randn(1, 3, 224, 224),)
        exported_program = convert_tflite_to_exported_program(
            tflite_model_path=mock_tflite_path,
            example_inputs=example_inputs
        )
        
        if exported_program is not None:
            print(f"\nExportedProgram type: {type(exported_program)}")
            print("✓ ExportedProgram created successfully!")
        else:
            print("\nNote: ExportedProgram creation returned None")
            print("(torch.export may not be available or export failed)")
    except Exception as e:
        print(f"\nNote: ExportedProgram creation: {e}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
