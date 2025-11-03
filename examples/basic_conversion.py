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
import tensorflow as tf
from tflite2torch import (
    convert_tflite_to_torch,
    convert_tflite_to_graph_module,
    convert_tflite_to_exported_program,
)


def create_example_tflite_model():
    """Create a simple TFLite model for demonstration."""
    # Create a simple Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,), name='input'),
        tf.keras.layers.Dense(8, activation='relu', name='dense1'),
        tf.keras.layers.Dense(5, activation='softmax', name='output')
    ])
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    return tflite_model


def main():
    """Main example function."""
    print("=" * 70)
    print("TFLite to PyTorch Conversion Example")
    print("=" * 70)
    print()
    
    # Create a real TFLite model for demonstration
    print("Creating example TFLite model...")
    tflite_model = create_example_tflite_model()
    
    tflite_path = "/tmp/example_model.tflite"
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"✓ Created TFLite model ({len(tflite_model)} bytes)")
    print(f"Converting TFLite model: {tflite_path}")
    print()
    
    # Method 1: Generate PyTorch code
    print("Method 1: Generate PyTorch code")
    print("-" * 70)
    code = convert_tflite_to_torch(
        tflite_model_path=tflite_path,
        output_path="/tmp/converted_model.py"
    )
    print("\nGenerated code (first 500 characters):")
    print(code[:500] + "..." if len(code) > 500 else code)
    print(f"\n✓ Code saved to: /tmp/converted_model.py")
    print()
    
    # Method 2: Convert to GraphModule for direct execution
    print("\n" + "=" * 70)
    print("Method 2: Convert to GraphModule")
    print("-" * 70)
    graph_module = convert_tflite_to_graph_module(
        tflite_model_path=tflite_path
    )
    
    print(f"\nGraphModule type: {type(graph_module)}")
    print(f"GraphModule: {graph_module}")
    
    # Test the model with sample input (matching the input shape of our model)
    print("\nTesting the converted model...")
    try:
        sample_input = torch.randn(1, 10)  # Input shape matches our model: (batch_size, 10)
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
        # Provide example inputs for export (matching our model's input shape)
        example_inputs = (torch.randn(1, 10),)
        exported_program = convert_tflite_to_exported_program(
            tflite_model_path=tflite_path,
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
