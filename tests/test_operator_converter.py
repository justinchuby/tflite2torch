"""Tests for operator converter module."""

import pytest
import torch
import torch.nn as nn
from tflite2torch.operator_converter import OperatorConverter


class TestOperatorConverter:
    """Tests for OperatorConverter class."""

    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = OperatorConverter()
        assert len(converter.converters) > 0

    def test_convert_conv2d(self):
        """Test CONV_2D conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "CONV_2D",
            inputs=[0, 1, 2],
            options={
                "stride_h": 1,
                "stride_w": 1,
                "padding": "SAME",
                "fused_activation_function": "RELU"
            }
        )
        assert result["module"] == nn.Conv2d
        assert result["params"]["stride"] == (1, 1)
        assert result["activation"] == "RELU"

    def test_convert_fully_connected(self):
        """Test FULLY_CONNECTED conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "FULLY_CONNECTED",
            inputs=[0, 1],
            options={}
        )
        assert result["module"] == nn.Linear

    def test_convert_relu(self):
        """Test RELU conversion."""
        converter = OperatorConverter()
        result = converter.convert("RELU", inputs=[0], options={})
        assert result["module"] == nn.ReLU

    def test_convert_add(self):
        """Test ADD conversion."""
        converter = OperatorConverter()
        result = converter.convert("ADD", inputs=[0, 1], options={})
        assert result["module"] == torch.add

    def test_convert_max_pool2d(self):
        """Test MAX_POOL_2D conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "MAX_POOL_2D",
            inputs=[0],
            options={
                "stride_h": 2,
                "stride_w": 2,
                "filter_height": 2,
                "filter_width": 2,
                "padding": "VALID"
            }
        )
        assert result["module"] == nn.MaxPool2d
        assert result["params"]["kernel_size"] == (2, 2)
        assert result["params"]["stride"] == (2, 2)

    def test_convert_softmax(self):
        """Test SOFTMAX conversion."""
        converter = OperatorConverter()
        result = converter.convert("SOFTMAX", inputs=[0], options={})
        assert result["module"] == nn.Softmax
        assert result["params"]["dim"] == -1

    def test_convert_reshape(self):
        """Test RESHAPE conversion."""
        converter = OperatorConverter()
        result = converter.convert("RESHAPE", inputs=[0, 1], options={})
        assert result["module"] == torch.reshape

    def test_convert_concatenation(self):
        """Test CONCATENATION conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "CONCATENATION",
            inputs=[0, 1],
            options={"axis": 1}
        )
        assert result["module"] == torch.cat
        assert result["params"]["dim"] == 1

    def test_convert_unsupported_operator(self):
        """Test conversion of unsupported operator."""
        converter = OperatorConverter()
        with pytest.raises(NotImplementedError):
            converter.convert("UNSUPPORTED_OP", inputs=[], options={})

    def test_get_activation_module_relu(self):
        """Test getting RELU activation module."""
        converter = OperatorConverter()
        activation = converter.get_activation_module("RELU")
        assert isinstance(activation, nn.ReLU)

    def test_get_activation_module_none(self):
        """Test getting NONE activation."""
        converter = OperatorConverter()
        activation = converter.get_activation_module("NONE")
        assert activation is None

    def test_get_activation_module_relu6(self):
        """Test getting RELU6 activation module."""
        converter = OperatorConverter()
        activation = converter.get_activation_module("RELU6")
        assert isinstance(activation, nn.ReLU6)

    def test_convert_depthwise_conv2d(self):
        """Test DEPTHWISE_CONV_2D conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "DEPTHWISE_CONV_2D",
            inputs=[0, 1],
            options={
                "stride_h": 1,
                "stride_w": 1,
                "padding": "SAME"
            }
        )
        assert result["module"] == nn.Conv2d
        assert result["depthwise"] is True

    def test_convert_transpose(self):
        """Test TRANSPOSE conversion."""
        converter = OperatorConverter()
        result = converter.convert("TRANSPOSE", inputs=[0, 1], options={})
        assert result["module"] == torch.permute

    def test_convert_mean(self):
        """Test MEAN conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "MEAN",
            inputs=[0, 1],
            options={"keep_dims": True}
        )
        assert result["module"] == torch.mean
        assert result["params"]["keepdim"] is True

    def test_convert_mul(self):
        """Test MUL conversion."""
        converter = OperatorConverter()
        result = converter.convert("MUL", inputs=[0, 1], options={})
        assert result["module"] == torch.mul

    def test_convert_sub(self):
        """Test SUB conversion."""
        converter = OperatorConverter()
        result = converter.convert("SUB", inputs=[0, 1], options={})
        assert result["module"] == torch.sub

    def test_convert_div(self):
        """Test DIV conversion."""
        converter = OperatorConverter()
        result = converter.convert("DIV", inputs=[0, 1], options={})
        assert result["module"] == torch.div
