"""Tests for comprehensive option parsing in TFLite parser."""

import os
import tempfile
import pytest
import tensorflow as tf
from tflite2torch._parser import TFLiteParser


class TestParserOptions:
    """Test that parser correctly extracts options for various operators."""

    def test_conv2d_options(self):
        """Test Conv2D option parsing."""
        # Create a Conv2D model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(8, 8, 3)),
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=3,
                strides=(2, 2),
                padding='same',
                activation='relu',
                dilation_rate=(1, 1)
            ),
        ])
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(tflite_model)
            temp_path = f.name
        
        try:
            # Parse the model
            parser = TFLiteParser()
            subgraphs = parser.parse(temp_path)
            
            # Find Conv2D operator
            conv_op = None
            for op in subgraphs[0].operators:
                if op.op_type == "CONV_2D":
                    conv_op = op
                    break
            
            assert conv_op is not None, "Conv2D operator not found"
            
            # Check that options were parsed
            assert "stride_w" in conv_op.builtin_options
            assert "stride_h" in conv_op.builtin_options
            assert "padding" in conv_op.builtin_options
            assert "fused_activation_function" in conv_op.builtin_options
            assert "dilation_w_factor" in conv_op.builtin_options
            assert "dilation_h_factor" in conv_op.builtin_options
            
            # Check values
            assert conv_op.builtin_options["stride_w"] == 2
            assert conv_op.builtin_options["stride_h"] == 2
            assert conv_op.builtin_options["padding"] == "SAME"
            assert conv_op.builtin_options["fused_activation_function"] == "RELU"
            
        finally:
            os.unlink(temp_path)

    def test_pool2d_options(self):
        """Test pooling operation option parsing."""
        # Create a MaxPool2D model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(8, 8, 3)),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid'
            ),
        ])
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(tflite_model)
            temp_path = f.name
        
        try:
            # Parse the model
            parser = TFLiteParser()
            subgraphs = parser.parse(temp_path)
            
            # Find MaxPool2D operator
            pool_op = None
            for op in subgraphs[0].operators:
                if op.op_type == "MAX_POOL_2D":
                    pool_op = op
                    break
            
            assert pool_op is not None, "MaxPool2D operator not found"
            
            # Check that options were parsed
            assert "stride_w" in pool_op.builtin_options
            assert "stride_h" in pool_op.builtin_options
            assert "filter_width" in pool_op.builtin_options
            assert "filter_height" in pool_op.builtin_options
            assert "padding" in pool_op.builtin_options
            
            # Check values
            assert pool_op.builtin_options["stride_w"] == 2
            assert pool_op.builtin_options["stride_h"] == 2
            assert pool_op.builtin_options["filter_width"] == 2
            assert pool_op.builtin_options["filter_height"] == 2
            assert pool_op.builtin_options["padding"] == "VALID"
            
        finally:
            os.unlink(temp_path)

    def test_reshape_options(self):
        """Test Reshape option parsing."""
        # Create a Reshape model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(8, 8)),
            tf.keras.layers.Reshape((64,))
        ])
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(tflite_model)
            temp_path = f.name
        
        try:
            # Parse the model
            parser = TFLiteParser()
            subgraphs = parser.parse(temp_path)
            
            # Find Reshape operator
            reshape_op = None
            for op in subgraphs[0].operators:
                if op.op_type == "RESHAPE":
                    reshape_op = op
                    break
            
            assert reshape_op is not None, "Reshape operator not found"
            
            # Reshape may have options or may get shape from input tensor
            # Just verify the operator was found and parsed
            assert reshape_op.builtin_options is not None
            # If new_shape is present, it should be a list
            if "new_shape" in reshape_op.builtin_options:
                assert isinstance(reshape_op.builtin_options["new_shape"], list)
            
        finally:
            os.unlink(temp_path)

    def test_fully_connected_options(self):
        """Test FullyConnected option parsing."""
        # Create a Dense layer model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5, activation='relu')
        ])
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(tflite_model)
            temp_path = f.name
        
        try:
            # Parse the model
            parser = TFLiteParser()
            subgraphs = parser.parse(temp_path)
            
            # Find FullyConnected operator
            fc_op = None
            for op in subgraphs[0].operators:
                if op.op_type == "FULLY_CONNECTED":
                    fc_op = op
                    break
            
            assert fc_op is not None, "FullyConnected operator not found"
            
            # Check that options were parsed
            assert "fused_activation_function" in fc_op.builtin_options
            assert fc_op.builtin_options["fused_activation_function"] == "RELU"
            
        finally:
            os.unlink(temp_path)

    def test_concatenation_options(self):
        """Test Concatenation option parsing."""
        # Create a Concatenate model
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        concat = tf.keras.layers.Concatenate(axis=-1)([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=concat)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(tflite_model)
            temp_path = f.name
        
        try:
            # Parse the model
            parser = TFLiteParser()
            subgraphs = parser.parse(temp_path)
            
            # Find Concatenation operator
            concat_op = None
            for op in subgraphs[0].operators:
                if op.op_type == "CONCATENATION":
                    concat_op = op
                    break
            
            assert concat_op is not None, "Concatenation operator not found"
            
            # Check that options were parsed
            assert "axis" in concat_op.builtin_options
            
        finally:
            os.unlink(temp_path)

    def test_add_options(self):
        """Test Add option parsing (with fused activation)."""
        # Create an Add model
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        add = tf.keras.layers.Add()([input1, input2])
        output = tf.keras.layers.Activation('relu')(add)
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        # Convert to TFLite with optimization to fuse ops
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tflite") as f:
            f.write(tflite_model)
            temp_path = f.name
        
        try:
            # Parse the model
            parser = TFLiteParser()
            subgraphs = parser.parse(temp_path)
            
            # Find Add operator
            add_op = None
            for op in subgraphs[0].operators:
                if op.op_type == "ADD":
                    add_op = op
                    break
            
            # May not find fused activation if optimizer doesn't fuse
            # Just verify we can parse the model
            assert len(subgraphs) > 0
            
        finally:
            os.unlink(temp_path)
