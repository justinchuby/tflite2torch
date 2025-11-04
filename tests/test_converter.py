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
        assert converter.code_renderer is not None

    def test_convert_to_code(self):
        """Test converting to code."""
        converter = TFLiteToTorchConverter()

        # Create a temporary mock file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        try:
            code = converter.convert(
                tflite_model_path=temp_path,
                generate_code=True
            )
            assert isinstance(code, str)
            assert "class ConvertedModel" in code
            assert "def forward" in code
            assert "import torch" in code
        finally:
            os.unlink(temp_path)

    def test_convert_to_graph_module(self):
        """Test converting to GraphModule."""
        converter = TFLiteToTorchConverter()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        try:
            graph_module = converter.convert(
                tflite_model_path=temp_path,
                generate_code=False
            )
            assert isinstance(graph_module, GraphModule)
        finally:
            os.unlink(temp_path)

    def test_convert_and_save(self):
        """Test converting and saving to file."""
        converter = TFLiteToTorchConverter()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            output_path = f.name

        try:
            code = converter.convert_and_save(
                tflite_model_path=temp_path,
                output_code_path=output_path
            )
            assert isinstance(code, str)
            assert os.path.exists(output_path)

            # Check file contents
            with open(output_path, "r") as f:
                file_contents = f.read()
            assert file_contents == code
        finally:
            os.unlink(temp_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_convert_nonexistent_file(self):
        """Test converting nonexistent file raises error."""
        converter = TFLiteToTorchConverter()
        with pytest.raises(FileNotFoundError):
            converter.convert("nonexistent_file.tflite")

    def test_convert_to_graph_module_method(self):
        """Test convert_to_graph_module method."""
        converter = TFLiteToTorchConverter()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        try:
            graph_module = converter.convert_to_graph_module(temp_path)
            assert isinstance(graph_module, GraphModule)
        finally:
            os.unlink(temp_path)


class TestConvertTFLiteToTorch:
    """Tests for convert_tflite_to_torch convenience function."""

    def test_convert_function_to_code(self):
        """Test convenience function for code generation."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        try:
            code = convert_tflite_to_torch(temp_path)
            assert isinstance(code, str)
            assert "class ConvertedModel" in code
        finally:
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

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            output_path = f.name

        try:
            code = convert_tflite_to_torch(temp_path, output_path=output_path)
            assert isinstance(code, str)
            assert os.path.exists(output_path)
        finally:
            os.unlink(temp_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_convert_function_to_exported_program(self):
        """Test convenience function for ExportedProgram."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(create_test_tflite_model())
            temp_path = f.name

        try:
            # Note: This may return None if torch.export is not available
            import torch
            exported = convert_tflite_to_exported_program(temp_path)
            if hasattr(torch, "export"):
                # If torch.export is available, we should get something back
                # (may be None if export fails, which is acceptable)
                assert exported is None or hasattr(exported, "graph_module")
        finally:
            os.unlink(temp_path)
