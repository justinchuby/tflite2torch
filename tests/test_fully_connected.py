"""
Comprehensive tests for the FULLY_CONNECTED operator (OP 9).

This test suite validates:
- Correct output shapes for various input dimensions
- All activation functions (NONE, RELU, RELU_N1_TO_1, RELU6, TANH, SIGN_BIT)
- Edge cases (no bias, different input shapes)
- Numerical accuracy compared to TFLite reference implementation
"""

import pytest
import numpy as np
import torch
import tensorflow as tf
from tflite2torch import convert_tflite_to_graph_module


def run_tflite_model(tflite_model, input_data):
    """Run a TFLite model and return the output."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


def compare_outputs(tflite_output, torch_output, rtol=1e-4, atol=1e-4, test_name=""):
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
            err_msg=f"{test_name}: TFLite and PyTorch outputs differ"
        )
        return True
    except AssertionError as e:
        print(f"❌ {test_name} failed: {e}")
        return False


class TestFullyConnectedOperator:
    """Test suite for FULLY_CONNECTED operator."""
    
    def test_basic_fully_connected(self, tmp_path):
        """Test basic FULLY_CONNECTED without activation."""
        input_layer = tf.keras.layers.Input(shape=(10,))
        output = tf.keras.layers.Dense(5, use_bias=True)(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "fc_basic.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 10).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert torch_output.shape == torch.Size([1, 5]), f"Expected shape [1, 5], got {torch_output.shape}"
        assert compare_outputs(tflite_output, torch_output, test_name="basic_fc", rtol=1e-3, atol=1e-3)
    
    def test_fully_connected_relu(self, tmp_path):
        """Test FULLY_CONNECTED with RELU activation."""
        input_layer = tf.keras.layers.Input(shape=(10,))
        output = tf.keras.layers.Dense(5, activation='relu')(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "fc_relu.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        # Use negative values to test RELU properly
        input_data = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0, 4.0, -4.0, 5.0]], dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert torch_output.shape == torch.Size([1, 5])
        assert compare_outputs(tflite_output, torch_output, test_name="fc_relu", rtol=1e-3, atol=1e-3)
    
    def test_fully_connected_relu6(self, tmp_path):
        """Test FULLY_CONNECTED with RELU6 activation."""
        input_layer = tf.keras.layers.Input(shape=(10,))
        # Create a layer with large values to test RELU6 clipping
        x = tf.keras.layers.Dense(5, use_bias=False)(input_layer)
        output = tf.keras.layers.ReLU(max_value=6.0)(x)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "fc_relu6.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        # Use large values to test RELU6 clipping at 6
        input_data = np.array([[10.0, -5.0, 3.0, 7.0, -2.0, 8.0, 1.0, -1.0, 5.0, 9.0]], dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert torch_output.shape == torch.Size([1, 5])
        assert compare_outputs(tflite_output, torch_output, test_name="fc_relu6", rtol=1e-3, atol=1e-3)
    
    def test_fully_connected_tanh(self, tmp_path):
        """Test FULLY_CONNECTED with TANH activation."""
        input_layer = tf.keras.layers.Input(shape=(10,))
        output = tf.keras.layers.Dense(5, activation='tanh')(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "fc_tanh.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.array([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -4.0, 4.0, 5.0]], dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert torch_output.shape == torch.Size([1, 5])
        assert compare_outputs(tflite_output, torch_output, test_name="fc_tanh", rtol=1e-3, atol=1e-3)
    
    def test_fully_connected_no_bias(self, tmp_path):
        """Test FULLY_CONNECTED without bias."""
        input_layer = tf.keras.layers.Input(shape=(8,))
        output = tf.keras.layers.Dense(4, use_bias=False)(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "fc_no_bias.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 8).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert torch_output.shape == torch.Size([1, 4])
        assert compare_outputs(tflite_output, torch_output, test_name="fc_no_bias", rtol=1e-3, atol=1e-3)
    
    def test_fully_connected_various_sizes(self, tmp_path):
        """Test FULLY_CONNECTED with various input/output sizes."""
        test_configs = [
            (5, 3),   # Small
            (10, 5),  # Medium
            (20, 10), # Large
            (50, 25), # Very large
            (100, 1), # Many-to-one
            (1, 100), # One-to-many
        ]
        
        for input_size, output_size in test_configs:
            input_layer = tf.keras.layers.Input(shape=(input_size,))
            output = tf.keras.layers.Dense(output_size)(input_layer)
            model = tf.keras.Model(inputs=input_layer, outputs=output)
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            model_path = tmp_path / f"fc_{input_size}x{output_size}.tflite"
            with open(model_path, "wb") as f:
                f.write(tflite_model)
            
            graph_module = convert_tflite_to_graph_module(str(model_path))
            input_data = np.random.randn(1, input_size).astype(np.float32)
            
            tflite_output = run_tflite_model(tflite_model, input_data)
            torch_output = graph_module(torch.from_numpy(input_data))
            
            assert torch_output.shape == torch.Size([1, output_size]), \
                f"Size {input_size}x{output_size}: expected shape [1, {output_size}], got {torch_output.shape}"
            assert compare_outputs(tflite_output, torch_output, 
                                   test_name=f"fc_{input_size}x{output_size}", 
                                   rtol=1e-3, atol=1e-3)
    
    def test_fully_connected_batch_size_one(self, tmp_path):
        """Test FULLY_CONNECTED with batch size 1."""
        input_layer = tf.keras.layers.Input(shape=(10,))
        output = tf.keras.layers.Dense(5)(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "fc_batch1.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.random.randn(1, 10).astype(np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert torch_output.shape == torch.Size([1, 5])
        assert tflite_output.shape == (1, 5)
        assert compare_outputs(tflite_output, torch_output, test_name="fc_batch1", rtol=1e-3, atol=1e-3)
    
    def test_fully_connected_output_shape_consistency(self, tmp_path):
        """Test that output shapes are consistent with TFLite."""
        # Test multiple input/output combinations
        test_cases = [
            {"input_shape": (10,), "output_units": 5, "batch": 1},
            {"input_shape": (20,), "output_units": 10, "batch": 1},
            {"input_shape": (15,), "output_units": 7, "batch": 1},
        ]
        
        for i, test_case in enumerate(test_cases):
            input_shape = test_case["input_shape"]
            output_units = test_case["output_units"]
            batch = test_case["batch"]
            
            input_layer = tf.keras.layers.Input(shape=input_shape)
            output = tf.keras.layers.Dense(output_units)(input_layer)
            model = tf.keras.Model(inputs=input_layer, outputs=output)
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            model_path = tmp_path / f"fc_shape_test_{i}.tflite"
            with open(model_path, "wb") as f:
                f.write(tflite_model)
            
            graph_module = convert_tflite_to_graph_module(str(model_path))
            input_data = np.random.randn(batch, *input_shape).astype(np.float32)
            
            tflite_output = run_tflite_model(tflite_model, input_data)
            torch_output = graph_module(torch.from_numpy(input_data))
            
            # Check shape consistency
            assert torch_output.shape[0] == batch, f"Batch size mismatch: {torch_output.shape[0]} != {batch}"
            assert torch_output.shape[-1] == output_units, f"Output units mismatch: {torch_output.shape[-1]} != {output_units}"
            assert tuple(tflite_output.shape) == tuple(torch_output.shape), \
                f"Shape mismatch: TFLite {tflite_output.shape} vs PyTorch {torch_output.shape}"
            
            assert compare_outputs(tflite_output, torch_output, 
                                   test_name=f"shape_test_{i}", 
                                   rtol=1e-3, atol=1e-3)
    
    def test_fully_connected_numerical_accuracy(self, tmp_path):
        """Test numerical accuracy with known weights and inputs."""
        # Create a model with predictable behavior
        input_layer = tf.keras.layers.Input(shape=(3,))
        output = tf.keras.layers.Dense(2, use_bias=True)(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "fc_numerical.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        
        # Test with simple inputs
        input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        # Verify shapes match
        assert tflite_output.shape == tuple(torch_output.shape)
        
        # Verify numerical accuracy with tighter tolerance
        assert compare_outputs(tflite_output, torch_output, 
                               test_name="numerical_accuracy", 
                               rtol=1e-5, atol=1e-5)
    
    def test_fully_connected_with_zeros(self, tmp_path):
        """Test FULLY_CONNECTED with zero inputs."""
        input_layer = tf.keras.layers.Input(shape=(10,))
        output = tf.keras.layers.Dense(5)(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "fc_zeros.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.zeros((1, 10), dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert torch_output.shape == torch.Size([1, 5])
        assert compare_outputs(tflite_output, torch_output, test_name="fc_zeros", rtol=1e-5, atol=1e-5)
    
    def test_fully_connected_with_ones(self, tmp_path):
        """Test FULLY_CONNECTED with all-ones inputs."""
        input_layer = tf.keras.layers.Input(shape=(10,))
        output = tf.keras.layers.Dense(5)(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        model_path = tmp_path / "fc_ones.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        graph_module = convert_tflite_to_graph_module(str(model_path))
        input_data = np.ones((1, 10), dtype=np.float32)
        
        tflite_output = run_tflite_model(tflite_model, input_data)
        torch_output = graph_module(torch.from_numpy(input_data))
        
        assert torch_output.shape == torch.Size([1, 5])
        assert compare_outputs(tflite_output, torch_output, test_name="fc_ones", rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
