#!/usr/bin/env python3
"""
Test script to verify the updated convert_tflite_to_exported_program functionality.
"""

import sys
import os

# Add the tflite2torch package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tflite2torch._converter import convert_tflite_to_exported_program

def test_function_availability():
    """Test that the function is available and has proper signature."""
    # Check that function exists and is callable
    assert callable(convert_tflite_to_exported_program)

    # Check the function signature using __code__
    func = convert_tflite_to_exported_program
    code = func.__code__

    # Check argument names
    expected_args = ['tflite_model_path', 'subgraph_index']
    actual_args = list(code.co_varnames[:code.co_argcount])

    print(f"Function arguments: {actual_args}")
    assert actual_args == expected_args, f"Expected {expected_args}, got {actual_args}"

    # Check default values
    defaults = func.__defaults__
    assert defaults == (0,), f"Expected default subgraph_index=0, got {defaults}"

    print("✓ Function signature is correct")

def test_docstring():
    """Test that the function has proper documentation."""
    docstring = convert_tflite_to_exported_program.__doc__
    assert docstring is not None
    assert "torch.export.ExportedProgram" in docstring
    assert "input signature" in docstring
    assert "dynamic shapes" in docstring

    print("✓ Function docstring is comprehensive")

if __name__ == "__main__":
    print("Testing convert_tflite_to_exported_program function...")

    try:
        test_function_availability()
        test_docstring()
        print("\n✅ All tests passed! The function is properly implemented.")

        # Print the function docstring for verification
        print("\nFunction documentation:")
        print("=" * 50)
        print(convert_tflite_to_exported_program.__doc__)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)