"""
Basic example of converting a TFLite model to PyTorch.

This example demonstrates the main conversion API and shows how to use
the generated PyTorch model.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tflite2torch import convert_tflite_to_torch


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
    
    # Method 1: Convert and get generated code as string
    print("Method 1: Generate PyTorch code")
    print("-" * 70)
    code = convert_tflite_to_torch(
        tflite_model_path=mock_tflite_path,
        output_path="/tmp/converted_model.py",
        generate_code=True
    )
    print("\nGenerated code:")
    print(code)
    print()
    
    # Method 2: Convert to GraphModule for direct execution
    print("\n" + "=" * 70)
    print("Method 2: Convert to GraphModule")
    print("-" * 70)
    graph_module = convert_tflite_to_torch(
        tflite_model_path=mock_tflite_path,
        generate_code=False
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
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
