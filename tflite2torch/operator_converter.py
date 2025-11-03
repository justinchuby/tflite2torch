"""
TFLite to Torch operator conversion module.

This module provides mappings and conversions from TFLite operators
to their PyTorch equivalents.
"""

from typing import Dict, List, Any, Callable, Optional
import torch
import torch.nn as nn


class OperatorConverter:
    """
    Converts TFLite operators to PyTorch equivalents.

    This class maintains a registry of conversion functions that map
    TFLite operators to PyTorch operations.
    """

    def __init__(self):
        self.converters: Dict[str, Callable] = {}
        self._register_converters()

    def _register_converters(self):
        """Register all operator converters."""
        self.converters["CONV_2D"] = self._convert_conv2d
        self.converters["DEPTHWISE_CONV_2D"] = self._convert_depthwise_conv2d
        self.converters["FULLY_CONNECTED"] = self._convert_fully_connected
        self.converters["ADD"] = self._convert_add
        self.converters["MUL"] = self._convert_mul
        self.converters["SUB"] = self._convert_sub
        self.converters["DIV"] = self._convert_div
        self.converters["RELU"] = self._convert_relu
        self.converters["RELU6"] = self._convert_relu6
        self.converters["TANH"] = self._convert_tanh
        self.converters["LOGISTIC"] = self._convert_sigmoid
        self.converters["SOFTMAX"] = self._convert_softmax
        self.converters["MAX_POOL_2D"] = self._convert_max_pool2d
        self.converters["AVERAGE_POOL_2D"] = self._convert_avg_pool2d
        self.converters["RESHAPE"] = self._convert_reshape
        self.converters["CONCATENATION"] = self._convert_concatenation
        self.converters["TRANSPOSE"] = self._convert_transpose
        self.converters["MEAN"] = self._convert_mean
        self.converters["PAD"] = self._convert_pad
        self.converters["SQUEEZE"] = self._convert_squeeze
        self.converters["EXPAND_DIMS"] = self._convert_expand_dims
        self.converters["SLICE"] = self._convert_slice
        self.converters["GATHER"] = self._convert_gather
        self.converters["SPLIT"] = self._convert_split
        self.converters["BATCH_TO_SPACE_ND"] = self._convert_batch_to_space
        self.converters["SPACE_TO_BATCH_ND"] = self._convert_space_to_batch
        self.converters["RESIZE_BILINEAR"] = self._convert_resize_bilinear
        self.converters["RESIZE_NEAREST_NEIGHBOR"] = self._convert_resize_nearest

    def convert(
        self, op_type: str, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert a TFLite operator to PyTorch.

        Args:
            op_type: TFLite operator type
            inputs: List of input specifications
            options: Operator-specific options

        Returns:
            Dictionary containing PyTorch operator information including:
            - 'module': PyTorch module class or function
            - 'params': Parameters for the module
            - 'forward_fn': Optional custom forward function
        """
        if op_type not in self.converters:
            raise NotImplementedError(f"Operator {op_type} is not supported yet")

        return self.converters[op_type](inputs, options)

    def _convert_conv2d(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite CONV_2D to PyTorch Conv2d."""
        # Extract parameters from options
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        padding = options.get("padding", "SAME")
        dilation_h = options.get("dilation_h_factor", 1)
        dilation_w = options.get("dilation_w_factor", 1)

        # Convert padding from TFLite to PyTorch format
        if padding == "SAME":
            # PyTorch doesn't directly support "SAME" padding
            # Will need to compute padding values
            padding_mode = "same"
        elif padding == "VALID":
            padding_mode = 0
        else:
            padding_mode = 0

        return {
            "module": nn.Conv2d,
            "params": {
                "stride": (stride_h, stride_w),
                "padding": padding_mode,
                "dilation": (dilation_h, dilation_w),
            },
            "activation": options.get("fused_activation_function", "NONE"),
        }

    def _convert_depthwise_conv2d(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite DEPTHWISE_CONV_2D to PyTorch depthwise Conv2d."""
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        padding = options.get("padding", "SAME")
        depth_multiplier = options.get("depth_multiplier", 1)

        if padding == "SAME":
            padding_mode = "same"
        else:
            padding_mode = 0

        return {
            "module": nn.Conv2d,
            "params": {
                "stride": (stride_h, stride_w),
                "padding": padding_mode,
                "groups": -1,  # Will be set to in_channels
            },
            "activation": options.get("fused_activation_function", "NONE"),
            "depthwise": True,
        }

    def _convert_fully_connected(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite FULLY_CONNECTED to PyTorch Linear."""
        return {
            "module": nn.Linear,
            "params": {},
            "activation": options.get("fused_activation_function", "NONE"),
        }

    def _convert_add(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite ADD to PyTorch addition."""
        return {
            "module": torch.add,
            "params": {},
            "activation": options.get("fused_activation_function", "NONE"),
        }

    def _convert_mul(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite MUL to PyTorch multiplication."""
        return {
            "module": torch.mul,
            "params": {},
            "activation": options.get("fused_activation_function", "NONE"),
        }

    def _convert_sub(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SUB to PyTorch subtraction."""
        return {
            "module": torch.sub,
            "params": {},
        }

    def _convert_div(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite DIV to PyTorch division."""
        return {
            "module": torch.div,
            "params": {},
        }

    def _convert_relu(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite RELU to PyTorch ReLU."""
        return {
            "module": nn.ReLU,
            "params": {},
        }

    def _convert_relu6(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite RELU6 to PyTorch ReLU6."""
        return {
            "module": nn.ReLU6,
            "params": {},
        }

    def _convert_tanh(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite TANH to PyTorch Tanh."""
        return {
            "module": nn.Tanh,
            "params": {},
        }

    def _convert_sigmoid(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite LOGISTIC to PyTorch Sigmoid."""
        return {
            "module": nn.Sigmoid,
            "params": {},
        }

    def _convert_softmax(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SOFTMAX to PyTorch Softmax."""
        return {
            "module": nn.Softmax,
            "params": {"dim": -1},
        }

    def _convert_max_pool2d(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite MAX_POOL_2D to PyTorch MaxPool2d."""
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        filter_height = options.get("filter_height", 2)
        filter_width = options.get("filter_width", 2)
        padding = options.get("padding", "VALID")

        if padding == "SAME":
            padding_mode = "same"
        else:
            padding_mode = 0

        return {
            "module": nn.MaxPool2d,
            "params": {
                "kernel_size": (filter_height, filter_width),
                "stride": (stride_h, stride_w),
                "padding": padding_mode,
            },
        }

    def _convert_avg_pool2d(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite AVERAGE_POOL_2D to PyTorch AvgPool2d."""
        stride_h = options.get("stride_h", 1)
        stride_w = options.get("stride_w", 1)
        filter_height = options.get("filter_height", 2)
        filter_width = options.get("filter_width", 2)
        padding = options.get("padding", "VALID")

        if padding == "SAME":
            padding_mode = "same"
        else:
            padding_mode = 0

        return {
            "module": nn.AvgPool2d,
            "params": {
                "kernel_size": (filter_height, filter_width),
                "stride": (stride_h, stride_w),
                "padding": padding_mode,
            },
        }

    def _convert_reshape(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite RESHAPE to PyTorch reshape."""
        return {
            "module": torch.reshape,
            "params": {},
        }

    def _convert_concatenation(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite CONCATENATION to PyTorch cat."""
        axis = options.get("axis", 0)
        return {
            "module": torch.cat,
            "params": {"dim": axis},
        }

    def _convert_transpose(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite TRANSPOSE to PyTorch permute."""
        return {
            "module": torch.permute,
            "params": {},
        }

    def _convert_mean(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite MEAN to PyTorch mean."""
        keep_dims = options.get("keep_dims", False)
        return {
            "module": torch.mean,
            "params": {"keepdim": keep_dims},
        }

    def _convert_pad(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite PAD to PyTorch pad."""
        return {
            "module": torch.nn.functional.pad,
            "params": {},
        }

    def _convert_squeeze(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SQUEEZE to PyTorch squeeze."""
        squeeze_dims = options.get("squeeze_dims", None)
        return {
            "module": torch.squeeze,
            "params": {"dim": squeeze_dims},
        }

    def _convert_expand_dims(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite EXPAND_DIMS to PyTorch unsqueeze."""
        return {
            "module": torch.unsqueeze,
            "params": {},
        }

    def _convert_slice(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SLICE to PyTorch slice indexing."""
        return {
            "module": torch.slice,
            "params": {},
        }

    def _convert_gather(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite GATHER to PyTorch gather."""
        axis = options.get("axis", 0)
        return {
            "module": torch.gather,
            "params": {"dim": axis},
        }

    def _convert_split(self, inputs: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TFLite SPLIT to PyTorch split."""
        num_splits = options.get("num_splits", 1)
        return {
            "module": torch.split,
            "params": {"split_size_or_sections": num_splits},
        }

    def _convert_batch_to_space(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite BATCH_TO_SPACE_ND to PyTorch operations."""
        return {
            "module": "batch_to_space",
            "params": {},
            "custom": True,
        }

    def _convert_space_to_batch(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite SPACE_TO_BATCH_ND to PyTorch operations."""
        return {
            "module": "space_to_batch",
            "params": {},
            "custom": True,
        }

    def _convert_resize_bilinear(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite RESIZE_BILINEAR to PyTorch interpolate."""
        align_corners = options.get("align_corners", False)
        return {
            "module": nn.functional.interpolate,
            "params": {
                "mode": "bilinear",
                "align_corners": align_corners,
            },
        }

    def _convert_resize_nearest(
        self, inputs: List[Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert TFLite RESIZE_NEAREST_NEIGHBOR to PyTorch interpolate."""
        return {
            "module": nn.functional.interpolate,
            "params": {
                "mode": "nearest",
            },
        }

    def get_activation_module(self, activation: str) -> Optional[nn.Module]:
        """
        Get PyTorch activation module for a TFLite fused activation.

        Args:
            activation: TFLite activation function name

        Returns:
            PyTorch activation module or None
        """
        activation_map = {
            "NONE": None,
            "RELU": nn.ReLU(),
            "RELU6": nn.ReLU6(),
            "TANH": nn.Tanh(),
            "SIGN_BIT": None,
        }
        return activation_map.get(activation, None)
