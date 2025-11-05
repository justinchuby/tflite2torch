"""Tests for main converter module."""

import os
import tempfile
import pytest
import tensorflow as tf
from torch.fx import GraphModule
from tflite2torch._converter import (
    TFLiteToTorchConverter,
    convert_tflite_to_torch,
    convert_tflite_to_graph_module,
    convert_tflite_to_exported_program,
)


def create_test_tflite_model():
    """Create a simple TFLite model for testing."""
    # Create a simple Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(5, activation='relu', name='dense1'),
    ])

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    return tflite_model


class TestTFLiteToTorchConverter:
    """Tests for TFLiteToTorchConverter class."""

    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = TFLiteToTorchConverter()
        assert converter.parser is not None
        assert converter.operator_converter is not None
        assert converter.fx_reconstructor is not None

    def test_convert_to_code(self):
        """Test converting to code."""
        converter = TFLiteToTorchConverter()

        # Create a temporary mock file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        try:
            # Convert returns a GraphModule, not code string
            graph_module = converter.convert(tflite_model_path=temp_path)
            assert isinstance(graph_module, GraphModule)
            # Check that the graph module has the expected structure
            assert hasattr(graph_module, 'graph')
            assert hasattr(graph_module, 'forward')
        finally:
            os.unlink(temp_path)

    def test_convert_to_graph_module(self):
        """Test converting to GraphModule."""
        converter = TFLiteToTorchConverter()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        try:
            # convert() now always returns a GraphModule
            graph_module = converter.convert(tflite_model_path=temp_path)
            assert isinstance(graph_module, GraphModule)
        finally:
            os.unlink(temp_path)

    def test_convert_and_save(self):
        """Test converting and saving to file."""
        converter = TFLiteToTorchConverter()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as output_dir:
            # Convert to GraphModule
            graph_module = converter.convert(tflite_model_path=temp_path)
            assert isinstance(graph_module, GraphModule)
            
            # Save using to_folder method
            graph_module.to_folder(output_dir)
            assert os.path.exists(os.path.join(output_dir, "module.py"))
        
        os.unlink(temp_path)

    def test_convert_nonexistent_file(self):
        """Test converting nonexistent file raises error."""
        converter = TFLiteToTorchConverter()
        with pytest.raises(FileNotFoundError):
            converter.convert("nonexistent_file.tflite")

    def test_convert_to_graph_module_method(self):
        """Test convert method returns a GraphModule."""
        converter = TFLiteToTorchConverter()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        try:
            # The convert() method is the main API
            graph_module = converter.convert(temp_path)
            assert isinstance(graph_module, GraphModule)
        finally:
            os.unlink(temp_path)


class TestConvertTFLiteToTorch:
    """Tests for convert_tflite_to_torch convenience function."""

    def test_convert_function_to_code(self):
        """Test convenience function for converting and saving to folder."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        with tempfile.TemporaryDirectory() as output_dir:
            # convert_tflite_to_torch now requires output_path
            convert_tflite_to_torch(temp_path, output_dir)
            # Check that output files were created
            assert os.path.exists(os.path.join(output_dir, "module.py"))
        
        os.unlink(temp_path)

    def test_convert_function_to_graph_module(self):
        """Test convenience function for GraphModule."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        try:
            graph_module = convert_tflite_to_graph_module(temp_path)
            assert isinstance(graph_module, GraphModule)
        finally:
            os.unlink(temp_path)

    def test_convert_function_with_output_path(self):
        """Test convenience function with output path."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        with tempfile.TemporaryDirectory() as output_dir:
            # convert_tflite_to_torch saves to a folder, not a single file
            convert_tflite_to_torch(temp_path, output_dir)
            # Check that output files were created
            assert os.path.exists(os.path.join(output_dir, "module.py"))
        
        os.unlink(temp_path)

    def test_convert_function_to_exported_program(self):
        """Test convenience function for ExportedProgram."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        try:
            # Note: torch.export is available in PyTorch 2.7+
            import torch
            if hasattr(torch, "export"):
                exported = convert_tflite_to_exported_program(temp_path)
                # Should return an ExportedProgram
                assert exported is not None
                assert hasattr(exported, "graph_module")
        finally:
            os.unlink(temp_path)
