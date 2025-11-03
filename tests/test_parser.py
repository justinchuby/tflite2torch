"""Tests for TFLite parser module."""

import os
import tempfile
import pytest
import tensorflow as tf
from tflite2torch._parser import TFLiteParser, TensorInfo, OperatorInfo, SubgraphInfo


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


class TestTensorInfo:
    """Tests for TensorInfo class."""

    def test_tensor_info_creation(self):
        """Test creating a TensorInfo object."""
        tensor = TensorInfo(
            name="test_tensor",
            shape=[1, 224, 224, 3],
            dtype="float32",
            index=0
        )
        assert tensor.name == "test_tensor"
        assert tensor.shape == [1, 224, 224, 3]
        assert tensor.dtype == "float32"
        assert tensor.index == 0

    def test_tensor_info_repr(self):
        """Test TensorInfo string representation."""
        tensor = TensorInfo(name="test", shape=[1, 2], dtype="int32", index=0)
        repr_str = repr(tensor)
        assert "test" in repr_str
        assert "int32" in repr_str


class TestOperatorInfo:
    """Tests for OperatorInfo class."""

    def test_operator_info_creation(self):
        """Test creating an OperatorInfo object."""
        op = OperatorInfo(
            op_type="CONV_2D",
            inputs=[0, 1, 2],
            outputs=[3],
            builtin_options={"padding": "SAME"}
        )
        assert op.op_type == "CONV_2D"
        assert op.inputs == [0, 1, 2]
        assert op.outputs == [3]
        assert op.builtin_options["padding"] == "SAME"

    def test_operator_info_repr(self):
        """Test OperatorInfo string representation."""
        op = OperatorInfo(op_type="ADD", inputs=[0, 1], outputs=[2])
        repr_str = repr(op)
        assert "ADD" in repr_str


class TestSubgraphInfo:
    """Tests for SubgraphInfo class."""

    def test_subgraph_info_creation(self):
        """Test creating a SubgraphInfo object."""
        tensors = [TensorInfo("t1", [1], "float32", 0)]
        operators = [OperatorInfo("ADD", [0], [1])]
        subgraph = SubgraphInfo(
            tensors=tensors,
            operators=operators,
            inputs=[0],
            outputs=[1],
            name="main"
        )
        assert subgraph.name == "main"
        assert len(subgraph.tensors) == 1
        assert len(subgraph.operators) == 1


class TestTFLiteParser:
    """Tests for TFLiteParser class."""

    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = TFLiteParser()
        assert parser.subgraphs == []
        assert parser.version == 0

    def test_parse_with_real_file(self):
        """Test parsing with a real TFLite model."""
        parser = TFLiteParser()
        
        # Create a real TFLite model
        tflite_model = create_test_tflite_model()
        
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(tflite_model)
            temp_path = f.name
        
        try:
            subgraphs = parser.parse(temp_path)
            assert len(subgraphs) > 0
            assert parser.version > 0
        finally:
            os.unlink(temp_path)

    def test_parse_invalid_file(self):
        """Test parsing with invalid file."""
        parser = TFLiteParser()
        
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(b"X")  # Too small
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                parser.parse(temp_path)
        finally:
            os.unlink(temp_path)

    def test_get_tensor_by_index(self):
        """Test getting tensor by index."""
        parser = TFLiteParser()
        
        # Create and parse a real TFLite model
        tflite_model = create_test_tflite_model()
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(tflite_model)
            temp_path = f.name
        
        try:
            parser.parse(temp_path)
            tensor = parser.get_tensor_by_index(0, 0)
            assert tensor.index == 0
            assert len(tensor.shape) > 0
        finally:
            os.unlink(temp_path)

    def test_get_input_tensors(self):
        """Test getting input tensors."""
        parser = TFLiteParser()
        
        # Create and parse a real TFLite model
        tflite_model = create_test_tflite_model()
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(tflite_model)
            temp_path = f.name
        
        try:
            parser.parse(temp_path)
            inputs = parser.get_input_tensors(0)
            assert len(inputs) >= 1
        finally:
            os.unlink(temp_path)

    def test_get_output_tensors(self):
        """Test getting output tensors."""
        parser = TFLiteParser()
        
        # Create and parse a real TFLite model
        tflite_model = create_test_tflite_model()
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(tflite_model)
            temp_path = f.name
        
        try:
            parser.parse(temp_path)
            outputs = parser.get_output_tensors(0)
            assert len(outputs) >= 1
        finally:
            os.unlink(temp_path)

    def test_operator_codes_mapping(self):
        """Test operator codes are properly defined."""
        assert TFLiteParser.OPERATOR_CODES[0] == "ADD"
        assert TFLiteParser.OPERATOR_CODES[3] == "CONV_2D"
        assert TFLiteParser.OPERATOR_CODES[9] == "FULLY_CONNECTED"

    def test_dtype_mapping(self):
        """Test dtype mapping is properly defined."""
        assert TFLiteParser.DTYPE_MAP[0] == "float32"
        assert TFLiteParser.DTYPE_MAP[2] == "int32"
        assert TFLiteParser.DTYPE_MAP[3] == "uint8"
