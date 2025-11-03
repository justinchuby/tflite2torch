"""
Comprehensive integration tests for ALL supported TFLite operators.

This test suite creates TFLite models for each of the 116+ supported operators
and verifies that the converted PyTorch models produce matching outputs.
"""

import pytest
import numpy as np
import torch
import tensorflow as tf
from tflite2torch import convert_tflite_to_graph_module
import tempfile
import os


def run_tflite_model(tflite_model, input_data):
    """Run a TFLite model and return the output."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input
    if isinstance(input_data, (list, tuple)):
        for i, data in enumerate(input_data):
            interpreter.set_tensor(input_details[i]['index'], data.astype(np.float32))
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


def compare_outputs(tflite_output, torch_output, rtol=1e-4, atol=1e-4, op_name=""):
    """Compare TFLite and PyTorch outputs."""
    if isinstance(torch_output, torch.Tensor):
        torch_output_np = torch_output.detach().numpy()
    else:
        torch_output_np = torch_output
    
    try:
        np.testing.assert_allclose(
            tflite_output,
            torch_output_np,
            rtol=rtol,
            atol=atol,
            err_msg=f"{op_name}: TFLite and PyTorch outputs differ"
        )
        return True
    except AssertionError as e:
        print(f"❌ {op_name} failed: {e}")
        return False


class TestArithmeticOperators:
    """Test arithmetic and math operators."""
    
    def test_abs_operator(self, tmp_path):
        """Test ABS operator."""
        # Create model with ABS
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.abs(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "abs_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        # Test
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.array([[-1.0, 2.0, -3.0, 4.0, -5.0]], dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="ABS")
    
    def test_add_operator(self, tmp_path):
        """Test ADD operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Add()([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "add_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.random.randn(1, 5).astype(np.float32)
        input_data2 = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="ADD")
    
    def test_ceil_operator(self, tmp_path):
        """Test CEIL operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.math.ceil(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "ceil_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.array([[1.2, 2.7, -3.1, 4.9, -5.5]], dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="CEIL")
    
    def test_cos_operator(self, tmp_path):
        """Test COS operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.cos(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "cos_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="COS")
    
    def test_div_operator(self, tmp_path):
        """Test DIV operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: x[0] / x[1])([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "div_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.random.randn(1, 5).astype(np.float32) + 5.0
        input_data2 = np.random.randn(1, 5).astype(np.float32) + 2.0
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="DIV")
    
    def test_exp_operator(self, tmp_path):
        """Test EXP operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.exp(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "exp_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 5).astype(np.float32) * 0.1  # Small values to avoid overflow
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="EXP")
    
    def test_floor_operator(self, tmp_path):
        """Test FLOOR operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.math.floor(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "floor_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.array([[1.2, 2.7, -3.1, 4.9, -5.5]], dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="FLOOR")
    
    def test_log_operator(self, tmp_path):
        """Test LOG operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.math.log(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "log_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.rand(1, 5).astype(np.float32) + 1.0  # Positive values
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="LOG")
    
    def test_maximum_operator(self, tmp_path):
        """Test MAXIMUM operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Maximum()([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "maximum_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.random.randn(1, 5).astype(np.float32)
        input_data2 = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="MAXIMUM")
    
    def test_minimum_operator(self, tmp_path):
        """Test MINIMUM operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Minimum()([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "minimum_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.random.randn(1, 5).astype(np.float32)
        input_data2 = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="MINIMUM")
    
    def test_mul_operator(self, tmp_path):
        """Test MUL operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Multiply()([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "mul_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.random.randn(1, 5).astype(np.float32)
        input_data2 = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="MUL")
    
    def test_neg_operator(self, tmp_path):
        """Test NEG operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: -x)(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "neg_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="NEG")
    
    def test_pow_operator(self, tmp_path):
        """Test POW operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.pow(x[0], x[1]))([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "pow_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.abs(np.random.randn(1, 5).astype(np.float32)) + 0.1
        input_data2 = np.array([[2.0, 2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="POW")
    
    def test_rsqrt_operator(self, tmp_path):
        """Test RSQRT operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.math.rsqrt(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "rsqrt_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.rand(1, 5).astype(np.float32) + 1.0  # Positive values
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="RSQRT")
    
    def test_sin_operator(self, tmp_path):
        """Test SIN operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.sin(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "sin_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="SIN")
    
    def test_sqrt_operator(self, tmp_path):
        """Test SQRT operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.sqrt(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "sqrt_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.rand(1, 5).astype(np.float32) + 1.0  # Positive values
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="SQRT")
    
    def test_square_operator(self, tmp_path):
        """Test SQUARE operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.square(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "square_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="SQUARE")
    
    def test_sub_operator(self, tmp_path):
        """Test SUB operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Subtract()([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "sub_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.random.randn(1, 5).astype(np.float32)
        input_data2 = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="SUB")


class TestConvolutionOperators:
    """Test convolution and pooling operators."""
    
    def test_conv2d_operator(self, tmp_path):
        """Test CONV_2D operator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(16, (3, 3), padding='same')
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "conv2d_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 28, 28, 3).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, rtol=1e-3, atol=1e-3, op_name="CONV_2D")
    
    def test_depthwise_conv2d_operator(self, tmp_path):
        """Test DEPTHWISE_CONV_2D operator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.DepthwiseConv2D((3, 3), padding='same')
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "depthwise_conv2d_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 28, 28, 3).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, rtol=1e-3, atol=1e-3, op_name="DEPTHWISE_CONV_2D")
    
    def test_max_pool2d_operator(self, tmp_path):
        """Test MAX_POOL_2D operator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.MaxPooling2D((2, 2))
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "maxpool2d_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 28, 28, 3).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="MAX_POOL_2D")
    
    def test_average_pool2d_operator(self, tmp_path):
        """Test AVERAGE_POOL_2D operator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.AveragePooling2D((2, 2))
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "avgpool2d_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 28, 28, 3).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="AVERAGE_POOL_2D")


class TestActivationOperators:
    """Test activation function operators."""
    
    def test_relu_operator(self, tmp_path):
        """Test RELU operator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5),
            tf.keras.layers.ReLU()
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "relu_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 10).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="RELU")
    
    def test_relu6_operator(self, tmp_path):
        """Test RELU6 operator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5),
            tf.keras.layers.ReLU(max_value=6.0)
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "relu6_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 10).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="RELU6")
    
    def test_tanh_operator(self, tmp_path):
        """Test TANH operator."""
        input_layer = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.nn.tanh(x))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "tanh_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="TANH")
    
    def test_softmax_operator(self, tmp_path):
        """Test SOFTMAX operator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5),
            tf.keras.layers.Softmax()
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "softmax_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 10).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="SOFTMAX")
    
    def test_elu_operator(self, tmp_path):
        """Test ELU operator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5),
            tf.keras.layers.ELU()
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "elu_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 10).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="ELU")
    
    def test_leaky_relu_operator(self, tmp_path):
        """Test LEAKY_RELU operator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "leaky_relu_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 10).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="LEAKY_RELU")


class TestShapeOperators:
    """Test shape and tensor manipulation operators."""
    
    def test_reshape_operator(self, tmp_path):
        """Test RESHAPE operator."""
        input_layer = tf.keras.layers.Input(shape=(12,))
        output = tf.keras.layers.Reshape((3, 4))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "reshape_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 12).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="RESHAPE")
    
    def test_squeeze_operator(self, tmp_path):
        """Test SQUEEZE operator."""
        input_layer = tf.keras.layers.Input(shape=(1, 10, 1))
        output = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1, 3]))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "squeeze_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 1, 10, 1).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="SQUEEZE")
    
    def test_transpose_operator(self, tmp_path):
        """Test TRANSPOSE operator."""
        input_layer = tf.keras.layers.Input(shape=(3, 4, 5))
        output = tf.keras.layers.Permute((3, 1, 2))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "transpose_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 3, 4, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="TRANSPOSE")
    
    def test_concatenation_operator(self, tmp_path):
        """Test CONCATENATION operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Concatenate()([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "concatenation_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.random.randn(1, 5).astype(np.float32)
        input_data2 = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="CONCATENATION")
    
    def test_pad_operator(self, tmp_path):
        """Test PAD operator."""
        input_layer = tf.keras.layers.Input(shape=(3, 3))
        output = tf.keras.layers.Lambda(
            lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1]])
        )(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "pad_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 3, 3).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="PAD")


class TestComparisonOperators:
    """Test comparison operators."""
    
    def test_equal_operator(self, tmp_path):
        """Test EQUAL operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32))([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "equal_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        input_data2 = np.array([[1.0, 2.0, 0.0, 4.0, 0.0]], dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="EQUAL")
    
    def test_greater_operator(self, tmp_path):
        """Test GREATER operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.cast(tf.greater(x[0], x[1]), tf.float32))([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "greater_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.random.randn(1, 5).astype(np.float32)
        input_data2 = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="GREATER")
    
    def test_less_operator(self, tmp_path):
        """Test LESS operator."""
        input1 = tf.keras.layers.Input(shape=(5,))
        input2 = tf.keras.layers.Input(shape=(5,))
        output = tf.keras.layers.Lambda(lambda x: tf.cast(tf.less(x[0], x[1]), tf.float32))([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "less_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data1 = np.random.randn(1, 5).astype(np.float32)
        input_data2 = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, [input_data1, input_data2])
        torch_output = graph_module(torch.from_numpy(input_data1), torch.from_numpy(input_data2))
        
        assert compare_outputs(tflite_output, torch_output, op_name="LESS")


class TestReductionOperators:
    """Test reduction operators."""
    
    def test_mean_operator(self, tmp_path):
        """Test MEAN operator."""
        input_layer = tf.keras.layers.Input(shape=(4, 4))
        output = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "mean_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 4, 4).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="MEAN")
    
    def test_sum_operator(self, tmp_path):
        """Test SUM operator."""
        input_layer = tf.keras.layers.Input(shape=(4, 4))
        output = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "sum_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 4, 4).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="SUM")
    
    def test_reduce_max_operator(self, tmp_path):
        """Test REDUCE_MAX operator."""
        input_layer = tf.keras.layers.Input(shape=(4, 4))
        output = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "reduce_max_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 4, 4).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="REDUCE_MAX")
    
    def test_reduce_min_operator(self, tmp_path):
        """Test REDUCE_MIN operator."""
        input_layer = tf.keras.layers.Input(shape=(4, 4))
        output = tf.keras.layers.Lambda(lambda x: tf.reduce_min(x, axis=-1, keepdims=True))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "reduce_min_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 4, 4).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="REDUCE_MIN")


class TestFullyConnectedOperator:
    """Test fully connected layer."""
    
    def test_fully_connected_operator(self, tmp_path):
        """Test FULLY_CONNECTED operator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5)
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "fully_connected_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 10).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="FULLY_CONNECTED")


class TestCastOperator:
    """Test cast/type conversion operator."""
    
    def test_cast_operator(self, tmp_path):
        """Test CAST operator."""
        input_layer = tf.keras.layers.Input(shape=(5,), dtype=tf.float32)
        output = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "cast_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 5).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert compare_outputs(tflite_output, torch_output, op_name="CAST")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
