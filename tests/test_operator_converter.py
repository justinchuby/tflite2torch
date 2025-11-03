"""Tests for operator converter module."""

import pytest
import torch
import torch.nn as nn
from tflite2torch._operator_converter import OperatorConverter


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
    
    # Arithmetic & Math Operations Tests
    def test_convert_abs(self):
        """Test ABS conversion."""
        converter = OperatorConverter()
        result = converter.convert("ABS", inputs=[0], options={})
        assert result["module"] == torch.abs
    
    def test_convert_add_n(self):
        """Test ADD_N conversion."""
        converter = OperatorConverter()
        result = converter.convert("ADD_N", inputs=[0, 1, 2], options={})
        assert result["module"] == torch.stack
    
    def test_convert_ceil(self):
        """Test CEIL conversion."""
        converter = OperatorConverter()
        result = converter.convert("CEIL", inputs=[0], options={})
        assert result["module"] == torch.ceil
    
    def test_convert_cos(self):
        """Test COS conversion."""
        converter = OperatorConverter()
        result = converter.convert("COS", inputs=[0], options={})
        assert result["module"] == torch.cos
    
    def test_convert_exp(self):
        """Test EXP conversion."""
        converter = OperatorConverter()
        result = converter.convert("EXP", inputs=[0], options={})
        assert result["module"] == torch.exp
    
    def test_convert_floor(self):
        """Test FLOOR conversion."""
        converter = OperatorConverter()
        result = converter.convert("FLOOR", inputs=[0], options={})
        assert result["module"] == torch.floor
    
    def test_convert_floor_div(self):
        """Test FLOOR_DIV conversion."""
        converter = OperatorConverter()
        result = converter.convert("FLOOR_DIV", inputs=[0, 1], options={})
        assert result["module"] == torch.floor_divide
    
    def test_convert_floor_mod(self):
        """Test FLOOR_MOD conversion."""
        converter = OperatorConverter()
        result = converter.convert("FLOOR_MOD", inputs=[0, 1], options={})
        assert result["module"] == torch.fmod
    
    def test_convert_log(self):
        """Test LOG conversion."""
        converter = OperatorConverter()
        result = converter.convert("LOG", inputs=[0], options={})
        assert result["module"] == torch.log
    
    def test_convert_maximum(self):
        """Test MAXIMUM conversion."""
        converter = OperatorConverter()
        result = converter.convert("MAXIMUM", inputs=[0, 1], options={})
        assert result["module"] == torch.maximum
    
    def test_convert_minimum(self):
        """Test MINIMUM conversion."""
        converter = OperatorConverter()
        result = converter.convert("MINIMUM", inputs=[0, 1], options={})
        assert result["module"] == torch.minimum
    
    def test_convert_neg(self):
        """Test NEG conversion."""
        converter = OperatorConverter()
        result = converter.convert("NEG", inputs=[0], options={})
        assert result["module"] == torch.neg
    
    def test_convert_pow(self):
        """Test POW conversion."""
        converter = OperatorConverter()
        result = converter.convert("POW", inputs=[0, 1], options={})
        assert result["module"] == torch.pow
    
    def test_convert_rsqrt(self):
        """Test RSQRT conversion."""
        converter = OperatorConverter()
        result = converter.convert("RSQRT", inputs=[0], options={})
        assert result["module"] == torch.rsqrt
    
    def test_convert_sin(self):
        """Test SIN conversion."""
        converter = OperatorConverter()
        result = converter.convert("SIN", inputs=[0], options={})
        assert result["module"] == torch.sin
    
    def test_convert_sqrt(self):
        """Test SQRT conversion."""
        converter = OperatorConverter()
        result = converter.convert("SQRT", inputs=[0], options={})
        assert result["module"] == torch.sqrt
    
    def test_convert_square(self):
        """Test SQUARE conversion."""
        converter = OperatorConverter()
        result = converter.convert("SQUARE", inputs=[0], options={})
        assert result["module"] == torch.square
    
    def test_convert_squared_difference(self):
        """Test SQUARED_DIFFERENCE conversion."""
        converter = OperatorConverter()
        result = converter.convert("SQUARED_DIFFERENCE", inputs=[0, 1], options={})
        # Should return a lambda or custom function
        assert callable(result["module"])
    
    # Convolution & Pooling Tests
    def test_convert_average_pool_2d(self):
        """Test AVERAGE_POOL_2D conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "AVERAGE_POOL_2D",
            inputs=[0],
            options={
                "stride_h": 2,
                "stride_w": 2,
                "filter_height": 2,
                "filter_width": 2,
                "padding": "VALID"
            }
        )
        assert result["module"] == nn.AvgPool2d
        assert result["params"]["kernel_size"] == (2, 2)
    
    def test_convert_conv_3d(self):
        """Test CONV_3D conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "CONV_3D",
            inputs=[0, 1, 2],
            options={
                "stride_d": 1,
                "stride_h": 1,
                "stride_w": 1,
                "padding": "SAME"
            }
        )
        assert result["module"] == nn.Conv3d
    
    def test_convert_transpose_conv(self):
        """Test TRANSPOSE_CONV conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "TRANSPOSE_CONV",
            inputs=[0, 1],
            options={}
        )
        assert result["module"] == nn.ConvTranspose2d
    
    def test_convert_batch_matmul(self):
        """Test BATCH_MATMUL conversion."""
        converter = OperatorConverter()
        result = converter.convert("BATCH_MATMUL", inputs=[0, 1], options={})
        assert result["module"] == torch.bmm
    
    # Activation Functions Tests
    def test_convert_elu(self):
        """Test ELU conversion."""
        converter = OperatorConverter()
        result = converter.convert("ELU", inputs=[0], options={})
        assert result["module"] == nn.ELU
    
    def test_convert_gelu(self):
        """Test GELU conversion."""
        converter = OperatorConverter()
        result = converter.convert("GELU", inputs=[0], options={})
        assert result["module"] == nn.GELU
    
    def test_convert_hard_swish(self):
        """Test HARD_SWISH conversion."""
        converter = OperatorConverter()
        result = converter.convert("HARD_SWISH", inputs=[0], options={})
        assert result["module"] == nn.Hardswish
    
    def test_convert_leaky_relu(self):
        """Test LEAKY_RELU conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "LEAKY_RELU",
            inputs=[0],
            options={"alpha": 0.1}
        )
        assert result["module"] == nn.LeakyReLU
        assert result["params"]["negative_slope"] == 0.1
    
    def test_convert_logistic(self):
        """Test LOGISTIC (Sigmoid) conversion."""
        converter = OperatorConverter()
        result = converter.convert("LOGISTIC", inputs=[0], options={})
        assert result["module"] == nn.Sigmoid
    
    def test_convert_log_softmax(self):
        """Test LOG_SOFTMAX conversion."""
        converter = OperatorConverter()
        result = converter.convert("LOG_SOFTMAX", inputs=[0], options={})
        assert result["module"] == nn.LogSoftmax
    
    def test_convert_prelu(self):
        """Test PRELU conversion."""
        converter = OperatorConverter()
        result = converter.convert("PRELU", inputs=[0, 1], options={})
        assert result["module"] == nn.PReLU
    
    def test_convert_relu6(self):
        """Test RELU6 conversion."""
        converter = OperatorConverter()
        result = converter.convert("RELU6", inputs=[0], options={})
        assert result["module"] == nn.ReLU6
    
    def test_convert_tanh(self):
        """Test TANH conversion."""
        converter = OperatorConverter()
        result = converter.convert("TANH", inputs=[0], options={})
        assert result["module"] == nn.Tanh
    
    # Normalization Tests
    def test_convert_l2_normalization(self):
        """Test L2_NORMALIZATION conversion."""
        converter = OperatorConverter()
        result = converter.convert("L2_NORMALIZATION", inputs=[0], options={})
        assert result["module"] == torch.nn.functional.normalize
    
    def test_convert_local_response_normalization(self):
        """Test LOCAL_RESPONSE_NORMALIZATION conversion."""
        converter = OperatorConverter()
        result = converter.convert("LOCAL_RESPONSE_NORMALIZATION", inputs=[0], options={})
        assert result["module"] == nn.LocalResponseNorm
    
    # Reduction Operations Tests
    def test_convert_reduce_max(self):
        """Test REDUCE_MAX conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "REDUCE_MAX",
            inputs=[0, 1],
            options={"keep_dims": True}
        )
        assert callable(result["module"])
    
    def test_convert_reduce_min(self):
        """Test REDUCE_MIN conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "REDUCE_MIN",
            inputs=[0, 1],
            options={"keep_dims": False}
        )
        assert callable(result["module"])
    
    def test_convert_reduce_prod(self):
        """Test REDUCE_PROD conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "REDUCE_PROD",
            inputs=[0, 1],
            options={}
        )
        assert callable(result["module"])
    
    def test_convert_reduce_any(self):
        """Test REDUCE_ANY conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "REDUCE_ANY",
            inputs=[0, 1],
            options={}
        )
        assert callable(result["module"])
    
    def test_convert_sum(self):
        """Test SUM conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "SUM",
            inputs=[0, 1],
            options={"keep_dims": True}
        )
        assert callable(result["module"])
    
    # Shape & Tensor Manipulation Tests
    def test_convert_batch_to_space_nd(self):
        """Test BATCH_TO_SPACE_ND conversion."""
        converter = OperatorConverter()
        result = converter.convert("BATCH_TO_SPACE_ND", inputs=[0, 1], options={})
        assert callable(result["module"])
    
    def test_convert_broadcast_to(self):
        """Test BROADCAST_TO conversion."""
        converter = OperatorConverter()
        result = converter.convert("BROADCAST_TO", inputs=[0, 1], options={})
        assert result["module"] == torch.broadcast_to
    
    def test_convert_depth_to_space(self):
        """Test DEPTH_TO_SPACE conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "DEPTH_TO_SPACE",
            inputs=[0],
            options={"block_size": 2}
        )
        assert result["module"] == nn.PixelShuffle
    
    def test_convert_expand_dims(self):
        """Test EXPAND_DIMS conversion."""
        converter = OperatorConverter()
        result = converter.convert("EXPAND_DIMS", inputs=[0, 1], options={})
        assert result["module"] == torch.unsqueeze
    
    def test_convert_fill(self):
        """Test FILL conversion."""
        converter = OperatorConverter()
        result = converter.convert("FILL", inputs=[0, 1], options={})
        assert result["module"] == torch.full
    
    def test_convert_gather(self):
        """Test GATHER conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "GATHER",
            inputs=[0, 1],
            options={"axis": 0}
        )
        assert result["module"] == torch.index_select
    
    def test_convert_pad(self):
        """Test PAD conversion."""
        converter = OperatorConverter()
        result = converter.convert("PAD", inputs=[0, 1], options={})
        assert result["module"] == torch.nn.functional.pad
    
    def test_convert_reverse(self):
        """Test REVERSE conversion."""
        converter = OperatorConverter()
        result = converter.convert("REVERSE", inputs=[0, 1], options={})
        assert result["module"] == torch.flip
    
    def test_convert_slice(self):
        """Test SLICE conversion."""
        converter = OperatorConverter()
        result = converter.convert("SLICE", inputs=[0, 1, 2], options={})
        assert callable(result["module"])
    
    def test_convert_space_to_batch_nd(self):
        """Test SPACE_TO_BATCH_ND conversion."""
        converter = OperatorConverter()
        result = converter.convert("SPACE_TO_BATCH_ND", inputs=[0, 1, 2], options={})
        assert callable(result["module"])
    
    def test_convert_split(self):
        """Test SPLIT conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "SPLIT",
            inputs=[0, 1],
            options={"num_splits": 2}
        )
        assert result["module"] == torch.split
    
    def test_convert_squeeze(self):
        """Test SQUEEZE conversion."""
        converter = OperatorConverter()
        result = converter.convert("SQUEEZE", inputs=[0], options={})
        assert result["module"] == torch.squeeze
    
    def test_convert_strided_slice(self):
        """Test STRIDED_SLICE conversion."""
        converter = OperatorConverter()
        result = converter.convert("STRIDED_SLICE", inputs=[0, 1, 2, 3], options={})
        assert callable(result["module"])
    
    def test_convert_tile(self):
        """Test TILE conversion."""
        converter = OperatorConverter()
        result = converter.convert("TILE", inputs=[0, 1], options={})
        assert result["module"] == torch.tile
    
    def test_convert_unpack(self):
        """Test UNPACK conversion."""
        converter = OperatorConverter()
        result = converter.convert(
            "UNPACK",
            inputs=[0],
            options={"num": 3, "axis": 0}
        )
        assert result["module"] == torch.unbind
    
    # Comparison Operations Tests
    def test_convert_equal(self):
        """Test EQUAL conversion."""
        converter = OperatorConverter()
        result = converter.convert("EQUAL", inputs=[0, 1], options={})
        assert result["module"] == torch.eq
    
    def test_convert_greater(self):
        """Test GREATER conversion."""
        converter = OperatorConverter()
        result = converter.convert("GREATER", inputs=[0, 1], options={})
        assert result["module"] == torch.gt
    
    def test_convert_greater_equal(self):
        """Test GREATER_EQUAL conversion."""
        converter = OperatorConverter()
        result = converter.convert("GREATER_EQUAL", inputs=[0, 1], options={})
        assert result["module"] == torch.ge
    
    def test_convert_less(self):
        """Test LESS conversion."""
        converter = OperatorConverter()
        result = converter.convert("LESS", inputs=[0, 1], options={})
        assert result["module"] == torch.lt
    
    def test_convert_less_equal(self):
        """Test LESS_EQUAL conversion."""
        converter = OperatorConverter()
        result = converter.convert("LESS_EQUAL", inputs=[0, 1], options={})
        assert result["module"] == torch.le
    
    def test_convert_not_equal(self):
        """Test NOT_EQUAL conversion."""
        converter = OperatorConverter()
        result = converter.convert("NOT_EQUAL", inputs=[0, 1], options={})
        assert result["module"] == torch.ne
    
    # Logical Operations Tests
    def test_convert_logical_and(self):
        """Test LOGICAL_AND conversion."""
        converter = OperatorConverter()
        result = converter.convert("LOGICAL_AND", inputs=[0, 1], options={})
        assert result["module"] == torch.logical_and
    
    def test_convert_logical_or(self):
        """Test LOGICAL_OR conversion."""
        converter = OperatorConverter()
        result = converter.convert("LOGICAL_OR", inputs=[0, 1], options={})
        assert result["module"] == torch.logical_or
    
    def test_convert_logical_not(self):
        """Test LOGICAL_NOT conversion."""
        converter = OperatorConverter()
        result = converter.convert("LOGICAL_NOT", inputs=[0], options={})
        assert result["module"] == torch.logical_not
    
    # Selection Operations Tests
    def test_convert_arg_max(self):
        """Test ARG_MAX conversion."""
        converter = OperatorConverter()
        result = converter.convert("ARG_MAX", inputs=[0, 1], options={})
        assert result["module"] == torch.argmax
    
    def test_convert_arg_min(self):
        """Test ARG_MIN conversion."""
        converter = OperatorConverter()
        result = converter.convert("ARG_MIN", inputs=[0, 1], options={})
        assert result["module"] == torch.argmin
    
    def test_convert_one_hot(self):
        """Test ONE_HOT conversion."""
        converter = OperatorConverter()
        result = converter.convert("ONE_HOT", inputs=[0, 1, 2, 3], options={})
        assert result["module"] == torch.nn.functional.one_hot
    
    def test_convert_select(self):
        """Test SELECT conversion."""
        converter = OperatorConverter()
        result = converter.convert("SELECT", inputs=[0, 1, 2], options={})
        assert result["module"] == torch.where
    
    def test_convert_top_k(self):
        """Test TOP_K conversion."""
        converter = OperatorConverter()
        result = converter.convert("TOP_K", inputs=[0, 1], options={})
        assert result["module"] == torch.topk
    
    # Quantization Tests
    def test_convert_quantize(self):
        """Test QUANTIZE conversion."""
        converter = OperatorConverter()
        result = converter.convert("QUANTIZE", inputs=[0], options={})
        assert callable(result["module"])
    
    def test_convert_dequantize(self):
        """Test DEQUANTIZE conversion."""
        converter = OperatorConverter()
        result = converter.convert("DEQUANTIZE", inputs=[0], options={})
        assert callable(result["module"])
    
    def test_convert_fake_quant(self):
        """Test FAKE_QUANT conversion."""
        converter = OperatorConverter()
        result = converter.convert("FAKE_QUANT", inputs=[0], options={})
        assert callable(result["module"])
    
    # Embedding & Lookup Tests
    def test_convert_embedding_lookup(self):
        """Test EMBEDDING_LOOKUP conversion."""
        converter = OperatorConverter()
        result = converter.convert("EMBEDDING_LOOKUP", inputs=[0, 1], options={})
        assert result["module"] == nn.Embedding
    
    def test_convert_hashtable_lookup(self):
        """Test HASHTABLE_LOOKUP conversion."""
        converter = OperatorConverter()
        result = converter.convert("HASHTABLE_LOOKUP", inputs=[0, 1, 2], options={})
        assert callable(result["module"])
    
    # Advanced Operations Tests
    def test_convert_cumsum(self):
        """Test CUMSUM conversion."""
        converter = OperatorConverter()
        result = converter.convert("CUMSUM", inputs=[0, 1], options={})
        assert result["module"] == torch.cumsum
    
    def test_convert_matrix_diag(self):
        """Test MATRIX_DIAG conversion."""
        converter = OperatorConverter()
        result = converter.convert("MATRIX_DIAG", inputs=[0], options={})
        assert result["module"] == torch.diag
    
    def test_convert_segment_sum(self):
        """Test SEGMENT_SUM conversion."""
        converter = OperatorConverter()
        result = converter.convert("SEGMENT_SUM", inputs=[0, 1], options={})
        assert callable(result["module"])
    
    def test_convert_where(self):
        """Test WHERE conversion."""
        converter = OperatorConverter()
        result = converter.convert("WHERE", inputs=[0, 1, 2], options={})
        assert result["module"] == torch.where
    
    def test_convert_zeros_like(self):
        """Test ZEROS_LIKE conversion."""
        converter = OperatorConverter()
        result = converter.convert("ZEROS_LIKE", inputs=[0], options={})
        assert result["module"] == torch.zeros_like
