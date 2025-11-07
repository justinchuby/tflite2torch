"""Integration tests comparing TFLite model outputs with converted PyTorch outputs."""

import numpy as np
import torch
import tensorflow as tf
from tflite2torch import convert_tflite_to_graph_module


def create_tflite_model_simple():
    """Create a simple TFLite model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,), name='input'),
        tf.keras.layers.Dense(8, activation='relu', name='dense1'),
        tf.keras.layers.Dense(5, activation='softmax', name='output')
    ])
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model


def create_tflite_model_conv():
    """Create a Conv2D TFLite model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 3), name='input'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(10, activation='softmax', name='output')
    ])
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model


def run_tflite_model(tflite_model, input_data):
    """Run a TFLite model and return the output."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Resize tensors if needed to match input batch size
    interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
    interpreter.allocate_tensors()
    
    # Set input
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


class TestIntegration:
    """Integration tests comparing TFLite and PyTorch outputs."""
    
    def test_simple_dense_model_comparison(self, tmp_path):
        """Test that converted simple dense model produces similar outputs to TFLite."""
        # Create TFLite model
        tflite_model = create_tflite_model_simple()
        
        # Save to file
        model_path = tmp_path / "model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        # Convert to PyTorch
        graph_module = convert_tflite_to_graph_module(str(model_path))
        
        # Create test input
        input_data = np.random.randn(1, 10).astype(np.float32)
        
        # Run TFLite model
        tflite_output = run_tflite_model(tflite_model, input_data)
        
        # Run PyTorch model
        torch_input = torch.from_numpy(input_data)
        torch_output = graph_module(torch_input)
        
        # Convert to numpy for comparison
        if isinstance(torch_output, torch.Tensor):
            torch_output_np = torch_output.detach().numpy()
        else:
            torch_output_np = torch_output
        
        # Compare outputs (should be very close)
        # Note: There may be small numerical differences due to implementation differences
        np.testing.assert_allclose(
            tflite_output, 
            torch_output_np, 
            rtol=1e-4, 
            atol=1e-4,
            err_msg="TFLite and PyTorch outputs differ significantly"
        )
    
    def test_simple_dense_model_shapes(self, tmp_path):
        """Test that output shapes match between TFLite and PyTorch."""
        # Create TFLite model
        tflite_model = create_tflite_model_simple()
        
        # Save to file
        model_path = tmp_path / "model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        # Convert to PyTorch
        graph_module = convert_tflite_to_graph_module(str(model_path))
        
        # Create test input
        input_data = np.random.randn(1, 10).astype(np.float32)
        
        # Run TFLite model
        tflite_output = run_tflite_model(tflite_model, input_data)
        
        # Run PyTorch model
        torch_input = torch.from_numpy(input_data)
        torch_output = graph_module(torch_input)
        
        # Check shapes match
        if isinstance(torch_output, torch.Tensor):
            torch_shape = torch_output.shape
        else:
            torch_shape = torch_output.shape
        
        assert tflite_output.shape == tuple(torch_shape), \
            f"Shape mismatch: TFLite {tflite_output.shape} vs PyTorch {torch_shape}"
    
    def test_batch_size_variation(self, tmp_path):
        """Test that model works with different batch sizes."""
        # Create TFLite model
        tflite_model = create_tflite_model_simple()
        
        # Save to file
        model_path = tmp_path / "model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        # Convert to PyTorch
        graph_module = convert_tflite_to_graph_module(str(model_path))
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            input_data = np.random.randn(batch_size, 10).astype(np.float32)
            
            # Run TFLite model
            tflite_output = run_tflite_model(tflite_model, input_data)
            
            # Run PyTorch model
            torch_input = torch.from_numpy(input_data)
            torch_output = graph_module(torch_input)
            
            if isinstance(torch_output, torch.Tensor):
                torch_output_np = torch_output.detach().numpy()
            else:
                torch_output_np = torch_output
            
            # Compare outputs
            np.testing.assert_allclose(
                tflite_output, 
                torch_output_np, 
                rtol=1e-4, 
                atol=1e-4,
                err_msg=f"Outputs differ for batch_size={batch_size}"
            )
    
    def test_conv_model_comparison(self, tmp_path):
        """Test that converted Conv2D model produces similar outputs to TFLite."""
        # Create TFLite model with Conv2D
        tflite_model = create_tflite_model_conv()
        
        # Save to file
        model_path = tmp_path / "conv_model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        # Convert to PyTorch
        graph_module = convert_tflite_to_graph_module(str(model_path))
        
        # Create test input (NHWC format for TFLite)
        input_data = np.random.randn(1, 28, 28, 3).astype(np.float32)
        
        # Run TFLite model
        tflite_output = run_tflite_model(tflite_model, input_data)
        
        # Run PyTorch model
        torch_input = torch.from_numpy(input_data)
        torch_output = graph_module(torch_input)
        
        if isinstance(torch_output, torch.Tensor):
            torch_output_np = torch_output.detach().numpy()
        else:
            torch_output_np = torch_output
        
        # Compare outputs
        np.testing.assert_allclose(
            tflite_output, 
            torch_output_np, 
            rtol=1e-3,  # Slightly higher tolerance for conv models
            atol=1e-3,
            err_msg="TFLite and PyTorch Conv2D outputs differ significantly"
        )
    
    def test_deterministic_outputs(self, tmp_path):
        """Test that running the model multiple times produces the same output."""
        # Create TFLite model
        tflite_model = create_tflite_model_simple()
        
        # Save to file
        model_path = tmp_path / "model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        # Convert to PyTorch
        graph_module = convert_tflite_to_graph_module(str(model_path))
        
        # Create test input
        input_data = np.random.randn(1, 10).astype(np.float32)
        torch_input = torch.from_numpy(input_data)
        
        # Run multiple times
        outputs = []
        for _ in range(3):
            output = graph_module(torch_input)
            if isinstance(output, torch.Tensor):
                outputs.append(output.detach().numpy())
            else:
                outputs.append(output)
        
        # All outputs should be identical
        for i in range(1, len(outputs)):
            np.testing.assert_array_equal(
                outputs[0], 
                outputs[i],
                err_msg=f"Output {i} differs from first output"
            )
    
    def test_zero_input(self, tmp_path):
        """Test model behavior with zero input."""
        # Create TFLite model
        tflite_model = create_tflite_model_simple()
        
        # Save to file
        model_path = tmp_path / "model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        # Convert to PyTorch
        graph_module = convert_tflite_to_graph_module(str(model_path))
        
        # Create zero input
        input_data = np.zeros((1, 10), dtype=np.float32)
        
        # Run TFLite model
        tflite_output = run_tflite_model(tflite_model, input_data)
        
        # Run PyTorch model
        torch_input = torch.from_numpy(input_data)
        torch_output = graph_module(torch_input)
        
        if isinstance(torch_output, torch.Tensor):
            torch_output_np = torch_output.detach().numpy()
        else:
            torch_output_np = torch_output
        
        # Compare outputs
        np.testing.assert_allclose(
            tflite_output, 
            torch_output_np, 
            rtol=1e-5, 
            atol=1e-5,
            err_msg="Zero input produces different outputs"
        )
    
    def test_ones_input(self, tmp_path):
        """Test model behavior with ones input."""
        # Create TFLite model
        tflite_model = create_tflite_model_simple()
        
        # Save to file
        model_path = tmp_path / "model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        # Convert to PyTorch
        graph_module = convert_tflite_to_graph_module(str(model_path))
        
        # Create ones input
        input_data = np.ones((1, 10), dtype=np.float32)
        
        # Run TFLite model
        tflite_output = run_tflite_model(tflite_model, input_data)
        
        # Run PyTorch model
        torch_input = torch.from_numpy(input_data)
        torch_output = graph_module(torch_input)
        
        if isinstance(torch_output, torch.Tensor):
            torch_output_np = torch_output.detach().numpy()
        else:
            torch_output_np = torch_output
        
        # Compare outputs
        np.testing.assert_allclose(
            tflite_output, 
            torch_output_np, 
            rtol=1e-5, 
            atol=1e-5,
            err_msg="Ones input produces different outputs"
        )
    
    def test_softmax_output_sums_to_one(self, tmp_path):
        """Test that softmax outputs sum to 1."""
        # Create TFLite model
        tflite_model = create_tflite_model_simple()
        
        # Save to file
        model_path = tmp_path / "model.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        # Convert to PyTorch
        graph_module = convert_tflite_to_graph_module(str(model_path))
        
        # Create test input
        input_data = np.random.randn(1, 10).astype(np.float32)
        torch_input = torch.from_numpy(input_data)
        
        # Run PyTorch model
        torch_output = graph_module(torch_input)
        
        if isinstance(torch_output, torch.Tensor):
            torch_output_np = torch_output.detach().numpy()
        else:
            torch_output_np = torch_output
        
        # Check that output sums to 1 (softmax property)
        output_sum = np.sum(torch_output_np, axis=-1)
        np.testing.assert_allclose(
            output_sum, 
            1.0, 
            rtol=1e-5,
            err_msg="Softmax output does not sum to 1"
        )
