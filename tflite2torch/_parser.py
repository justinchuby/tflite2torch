"""
TFLite graph parsing module.

This module provides functionality to parse TFLite model files and extract
the computational graph structure, operators, tensors, and metadata.
"""

from typing import Any
import numpy as np

from tensorflow.lite.python import schema_py_generated as schema_fb


class TensorInfo:
    """Represents a tensor in the TFLite model."""

    def __init__(
        self,
        name: str,
        shape: list[int],
        dtype: str,
        index: int,
        quantization: dict[str, Any] | None = None,
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.index = index
        self.quantization = quantization or {}

    def __repr__(self):
        return f"TensorInfo(name={self.name}, shape={self.shape}, dtype={self.dtype})"


class OperatorInfo:
    """Represents an operator/node in the TFLite model."""

    def __init__(
        self,
        op_type: str,
        inputs: list[int],
        outputs: list[int],
        builtin_options: dict[str, Any] | None = None,
        custom_options: bytes | None = None,
    ):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.builtin_options = builtin_options or {}
        self.custom_options = custom_options

    def __repr__(self):
        return f"OperatorInfo(op_type={self.op_type}, inputs={self.inputs}, outputs={self.outputs})"


class SubgraphInfo:
    """Represents a subgraph in the TFLite model."""

    def __init__(
        self,
        tensors: list[TensorInfo],
        operators: list[OperatorInfo],
        inputs: list[int],
        outputs: list[int],
        name: str = "",
    ):
        self.tensors = tensors
        self.operators = operators
        self.inputs = inputs
        self.outputs = outputs
        self.name = name

    def __repr__(self):
        return (
            f"SubgraphInfo(name={self.name}, "
            f"num_tensors={len(self.tensors)}, "
            f"num_operators={len(self.operators)})"
        )


class TFLiteParser:
    """
    Parser for TFLite model files.

    This class handles the parsing of TFLite FlatBuffer format and extracts
    the model structure including tensors, operators, and subgraphs.
    """

    # Complete TFLite operator codes (code -> name) - all builtin operators from TFLite schema
    OPERATOR_CODES = {
        0: "ADD",
        1: "AVERAGE_POOL_2D",
        2: "CONCATENATION",
        3: "CONV_2D",
        4: "DEPTHWISE_CONV_2D",
        5: "DEPTH_TO_SPACE",
        6: "DEQUANTIZE",
        7: "EMBEDDING_LOOKUP",
        8: "FLOOR",
        9: "FULLY_CONNECTED",
        10: "HASHTABLE_LOOKUP",
        11: "L2_NORMALIZATION",
        12: "L2_POOL_2D",
        13: "LOCAL_RESPONSE_NORMALIZATION",
        14: "LOGISTIC",
        15: "LSH_PROJECTION",
        16: "LSTM",
        17: "MAX_POOL_2D",
        18: "MUL",
        19: "RELU",
        20: "RELU_N1_TO_1",
        21: "RELU6",
        22: "RESHAPE",
        23: "RESIZE_BILINEAR",
        24: "RNN",
        25: "SOFTMAX",
        26: "SPACE_TO_DEPTH",
        27: "SVDF",
        28: "TANH",
        29: "CONCAT_EMBEDDINGS",
        30: "SKIP_GRAM",
        31: "CALL",
        32: "CUSTOM",
        33: "EMBEDDING_LOOKUP_SPARSE",
        34: "PAD",
        35: "UNIDIRECTIONAL_SEQUENCE_RNN",
        36: "GATHER",
        37: "BATCH_TO_SPACE_ND",
        38: "SPACE_TO_BATCH_ND",
        39: "TRANSPOSE",
        40: "MEAN",
        41: "SUB",
        42: "DIV",
        43: "SQUEEZE",
        44: "UNIDIRECTIONAL_SEQUENCE_LSTM",
        45: "STRIDED_SLICE",
        46: "BIDIRECTIONAL_SEQUENCE_RNN",
        47: "EXP",
        48: "TOPK_V2",
        49: "SPLIT",
        50: "LOG_SOFTMAX",
        51: "DELEGATE",
        52: "BIDIRECTIONAL_SEQUENCE_LSTM",
        53: "CAST",
        54: "PRELU",
        55: "MAXIMUM",
        56: "ARG_MAX",
        57: "MINIMUM",
        58: "LESS",
        59: "NEG",
        60: "PADV2",
        61: "GREATER",
        62: "GREATER_EQUAL",
        63: "LESS_EQUAL",
        64: "SELECT",
        65: "SLICE",
        66: "SIN",
        67: "TRANSPOSE_CONV",
        68: "SPARSE_TO_DENSE",
        69: "TILE",
        70: "EXPAND_DIMS",
        71: "EQUAL",
        72: "NOT_EQUAL",
        73: "LOG",
        74: "SUM",
        75: "SQRT",
        76: "RSQRT",
        77: "SHAPE",
        78: "POW",
        79: "ARG_MIN",
        80: "FAKE_QUANT",
        81: "REDUCE_PROD",
        82: "REDUCE_MAX",
        83: "PACK",
        84: "LOGICAL_OR",
        85: "ONE_HOT",
        86: "LOGICAL_AND",
        87: "LOGICAL_NOT",
        88: "UNPACK",
        89: "REDUCE_MIN",
        90: "FLOOR_DIV",
        91: "REDUCE_ANY",
        92: "SQUARE",
        93: "ZEROS_LIKE",
        94: "FILL",
        95: "FLOOR_MOD",
        96: "RANGE",
        97: "RESIZE_NEAREST_NEIGHBOR",
        98: "LEAKY_RELU",
        99: "SQUARED_DIFFERENCE",
        100: "MIRROR_PAD",
        101: "ABS",
        102: "SPLIT_V",
        103: "UNIQUE",
        104: "CEIL",
        105: "REVERSE_V2",
        106: "ADD_N",
        107: "GATHER_ND",
        108: "COS",
        109: "WHERE",
        110: "RANK",
        111: "ELU",
        112: "REVERSE_SEQUENCE",
        113: "MATRIX_DIAG",
        114: "QUANTIZE",
        115: "MATRIX_SET_DIAG",
        116: "ROUND",
        117: "HARD_SWISH",
        118: "IF",
        119: "WHILE",
        120: "NON_MAX_SUPPRESSION_V4",
        121: "NON_MAX_SUPPRESSION_V5",
        122: "SCATTER_ND",
        123: "SELECT_V2",
        124: "DENSIFY",
        125: "SEGMENT_SUM",
        126: "BATCH_MATMUL",
        127: "PLACEHOLDER_FOR_GREATER_OP_CODES",
        128: "CUMSUM",
        129: "CALL_ONCE",
        130: "BROADCAST_TO",
        131: "RFFT2D",
        132: "CONV_3D",
        133: "IMAG",
        134: "REAL",
        135: "COMPLEX_ABS",
        136: "HASHTABLE",
        137: "HASHTABLE_FIND",
        138: "HASHTABLE_IMPORT",
        139: "HASHTABLE_SIZE",
        140: "REDUCE_ALL",
        141: "CONV_3D_TRANSPOSE",
        142: "VAR_HANDLE",
        143: "READ_VARIABLE",
        144: "ASSIGN_VARIABLE",
        145: "BROADCAST_ARGS",
        146: "RANDOM_STANDARD_NORMAL",
        147: "BUCKETIZE",
        148: "RANDOM_UNIFORM",
        149: "MULTINOMIAL",
        150: "GELU",
        151: "DYNAMIC_UPDATE_SLICE",
        152: "RELU_0_TO_1",
        153: "UNSORTED_SEGMENT_PROD",
        154: "UNSORTED_SEGMENT_MAX",
        155: "UNSORTED_SEGMENT_SUM",
        156: "ATAN2",
        157: "UNSORTED_SEGMENT_MIN",
        158: "SIGN",
        159: "BITCAST",
        160: "BITWISE_XOR",
        161: "RIGHT_SHIFT",
        162: "STABLEHLO_LOGISTIC",
        163: "STABLEHLO_ADD",
        164: "STABLEHLO_DIVIDE",
        165: "STABLEHLO_MULTIPLY",
        166: "STABLEHLO_MAXIMUM",
        167: "STABLEHLO_RESHAPE",
        168: "STABLEHLO_CLAMP",
        169: "STABLEHLO_CONCATENATE",
        170: "STABLEHLO_BROADCAST_IN_DIM",
        171: "STABLEHLO_CONVOLUTION",
        172: "STABLEHLO_SLICE",
        173: "STABLEHLO_CUSTOM_CALL",
        174: "STABLEHLO_REDUCE",
        175: "STABLEHLO_ABS",
        176: "STABLEHLO_AND",
        177: "STABLEHLO_COSINE",
        178: "STABLEHLO_EXPONENTIAL",
        179: "STABLEHLO_FLOOR",
        180: "STABLEHLO_LOG",
        181: "STABLEHLO_MINIMUM",
        182: "STABLEHLO_NEGATE",
        183: "STABLEHLO_OR",
        184: "STABLEHLO_POWER",
        185: "STABLEHLO_REMAINDER",
        186: "STABLEHLO_RSQRT",
        187: "STABLEHLO_SELECT",
        188: "STABLEHLO_SUBTRACT",
        189: "STABLEHLO_TANH",
        190: "STABLEHLO_SCATTER",
        191: "STABLEHLO_COMPARE",
        192: "STABLEHLO_CONVERT",
        193: "STABLEHLO_DYNAMIC_SLICE",
        194: "STABLEHLO_DYNAMIC_UPDATE_SLICE",
        195: "STABLEHLO_PAD",
        196: "STABLEHLO_IOTA",
        197: "STABLEHLO_DOT_GENERAL",
        198: "STABLEHLO_REDUCE_WINDOW",
        199: "STABLEHLO_SORT",
        200: "STABLEHLO_WHILE",
        201: "STABLEHLO_GATHER",
        202: "STABLEHLO_TRANSPOSE",
        203: "DILATE",
        204: "STABLEHLO_RNG_BIT_GENERATOR",
        205: "REDUCE_WINDOW",
        206: "STABLEHLO_COMPOSITE",
        207: "STABLEHLO_SHIFT_LEFT",
        208: "STABLEHLO_CBRT",
        209: "STABLEHLO_CASE",
    }

    # Reverse mapping (name -> code) for lookups
    OPCODE_MAP = {name: code for code, name in OPERATOR_CODES.items()}

    # TFLite data type mapping
    DTYPE_MAP = {
        0: "float32",
        1: "float16",
        2: "int32",
        3: "uint8",
        4: "int64",
        5: "string",
        6: "bool",
        7: "int16",
        8: "complex64",
        9: "int8",
        10: "float64",
        11: "complex128",
    }

    def __init__(self):
        self.subgraphs: list[SubgraphInfo] = []
        self.model_description: str = ""
        self.version: int = 0
        self.weights: dict[int, dict[int, np.ndarray]] = {}  # subgraph_idx -> tensor_idx -> weights

    def parse(self, model_path: str) -> list[SubgraphInfo]:
        """
        Parse a TFLite model file.

        Args:
            model_path: Path to the TFLite model file

        Returns:
            list of SubgraphInfo objects representing the model's subgraphs
        """
        # Read and validate the model file
        with open(model_path, "rb") as f:
            model_data = f.read()

        # Check for TFLite magic number
        if len(model_data) < 4:
            raise ValueError("Invalid TFLite model file: too small")

        # Parse using official TFLite schema
        self._parse_tflite_model(model_data)

        return self.subgraphs

    def get_weights(self, subgraph_idx: int = 0) -> dict[int, np.ndarray]:
        """
        Get weights for a specific subgraph.

        Args:
            subgraph_idx: Index of the subgraph

        Returns:
            Dictionary mapping tensor index to weight tensor (as numpy array)
        """
        return self.weights.get(subgraph_idx, {})

    def _parse_tflite_model(self, model_data: bytes):
        """
        Parse actual TFLite model using the official schema.

        Args:
            model_data: Raw bytes of the TFLite model file
        """
        # Parse the model using FlatBuffers
        model = schema_fb.Model.GetRootAsModel(model_data, 0)

        # Get model version and description
        self.version = model.Version()
        description = model.Description()
        self.model_description = description.decode("utf-8") if description else ""

        # Parse all subgraphs
        self.subgraphs = []
        self.weights = {}
        for subgraph_idx in range(model.SubgraphsLength()):
            subgraph = model.Subgraphs(subgraph_idx)
            self.subgraphs.append(self._parse_subgraph(subgraph, model))
            # Extract weights for this subgraph
            self.weights[subgraph_idx] = self._extract_weights(subgraph, model)

    def _parse_subgraph(self, subgraph, model) -> SubgraphInfo:
        """
        Parse a single subgraph from the TFLite model.

        Args:
            subgraph: TFLite subgraph object
            model: TFLite model object (for accessing buffers)

        Returns:
            SubgraphInfo object
        """
        # Parse tensors
        tensors = []
        for tensor_idx in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(tensor_idx)
            tensors.append(self._parse_tensor(tensor, tensor_idx, model))

        # Parse operators
        operators = []
        for op_idx in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(op_idx)
            operators.append(self._parse_operator(operator, model))

        # Get input and output indices
        inputs = [subgraph.Inputs(i) for i in range(subgraph.InputsLength())]
        outputs = [subgraph.Outputs(i) for i in range(subgraph.OutputsLength())]

        # Get subgraph name
        name_bytes = subgraph.Name()
        name = name_bytes.decode("utf-8") if name_bytes else f"subgraph_{len(self.subgraphs)}"

        return SubgraphInfo(
            tensors=tensors, operators=operators, inputs=inputs, outputs=outputs, name=name
        )

    def _parse_tensor(self, tensor, tensor_idx: int, model) -> TensorInfo:
        """
        Parse a tensor from the TFLite model.

        Args:
            tensor: TFLite tensor object
            tensor_idx: Index of the tensor in the subgraph
            model: TFLite model object

        Returns:
            TensorInfo object
        """
        # Get tensor name
        name_bytes = tensor.Name()
        name = name_bytes.decode("utf-8") if name_bytes else f"tensor_{tensor_idx}"

        # Get tensor shape
        shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]

        # Get tensor dtype
        dtype_code = tensor.Type()
        dtype = TFLiteParser.DTYPE_MAP.get(dtype_code, "unknown")

        # Parse quantization parameters
        quantization = {}
        quant = tensor.Quantization()
        if quant:
            if quant.ScaleLength() > 0:
                quantization["scale"] = [quant.Scale(i) for i in range(quant.ScaleLength())]
            if quant.ZeroPointLength() > 0:
                quantization["zero_point"] = [
                    quant.ZeroPoint(i) for i in range(quant.ZeroPointLength())
                ]

        return TensorInfo(
            name=name, shape=shape, dtype=dtype, index=tensor_idx, quantization=quantization
        )

    def _parse_operator(self, operator, model) -> OperatorInfo:
        """
        Parse an operator from the TFLite model.

        Args:
            operator: TFLite operator object
            model: TFLite model object

        Returns:
            OperatorInfo object
        """
        # Get operator code
        opcode_index = operator.OpcodeIndex()
        opcode = model.OperatorCodes(opcode_index)

        # Get builtin code
        builtin_code = opcode.BuiltinCode()

        # Map builtin code to operator name
        op_type = self._get_operator_name(builtin_code, opcode)

        # Get inputs and outputs
        inputs = [operator.Inputs(i) for i in range(operator.InputsLength())]
        outputs = [operator.Outputs(i) for i in range(operator.OutputsLength())]

        # Parse builtin options
        builtin_options = self._parse_builtin_options(operator, builtin_code)

        # Get custom options
        custom_options = None
        if operator.CustomOptionsLength() > 0:
            custom_options = bytes(
                [operator.CustomOptions(i) for i in range(operator.CustomOptionsLength())]
            )

        return OperatorInfo(
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            builtin_options=builtin_options,
            custom_options=custom_options,
        )

    def _get_operator_name(self, builtin_code: int, opcode) -> str:
        """
        Get the operator name from builtin code.

        Args:
            builtin_code: TFLite builtin code
            opcode: TFLite opcode object

        Returns:
            Operator name as string
        """
        # Lookup in OPERATOR_CODES
        if builtin_code in TFLiteParser.OPERATOR_CODES:
            return TFLiteParser.OPERATOR_CODES[builtin_code]

        # If custom operator, try to get custom code
        custom_code = opcode.CustomCode()
        if custom_code:
            return custom_code.decode("utf-8")

        return f"UNKNOWN_{builtin_code}"

    def _parse_builtin_options(self, operator, builtin_code: int) -> dict[str, Any]:
        """
        Parse builtin options for an operator.

        Args:
            operator: TFLite operator object
            builtin_code: Builtin operator code

        Returns:
            Dictionary of parsed options
        """
        options = {}
        builtin_options_type = operator.BuiltinOptionsType()

        if builtin_options_type == 0:  # NONE
            return options

        # Get the builtin options object
        builtin_opts = operator.BuiltinOptions()
        if not builtin_opts:
            return options

        # Parse common options based on operator type
        # This is a simplified version - full implementation would handle all operator types
        try:
            # Try to parse as Conv2DOptions (most common)
            if builtin_code == TFLiteParser.OPCODE_MAP.get("CONV_2D"):
                opts = schema_fb.Conv2DOptions()
                opts.Init(builtin_opts.Bytes, builtin_opts.Pos)
                options["padding"] = self._get_padding_name(opts.Padding())
                options["stride_w"] = opts.StrideW()
                options["stride_h"] = opts.StrideH()
                options["fused_activation_function"] = self._get_activation_name(
                    opts.FusedActivationFunction()
                )
            elif builtin_code == TFLiteParser.OPCODE_MAP.get("FULLY_CONNECTED"):
                opts = schema_fb.FullyConnectedOptions()
                opts.Init(builtin_opts.Bytes, builtin_opts.Pos)
                options["fused_activation_function"] = self._get_activation_name(
                    opts.FusedActivationFunction()
                )
            elif builtin_code == TFLiteParser.OPCODE_MAP.get("LEAKY_RELU"):
                opts = schema_fb.LeakyReluOptions()
                opts.Init(builtin_opts.Bytes, builtin_opts.Pos)
                options["alpha"] = opts.Alpha()
            elif builtin_code == TFLiteParser.OPCODE_MAP.get("CONV_3D"):
                opts = schema_fb.Conv3DOptions()
                opts.Init(builtin_opts.Bytes, builtin_opts.Pos)
                options["padding"] = self._get_padding_name(opts.Padding())
                options["stride"] = [opts.StrideD(), opts.StrideH(), opts.StrideW()]
                options["fused_activation_function"] = self._get_activation_name(
                    opts.FusedActivationFunction()
                )
            # Add more operator-specific option parsing as needed
        except Exception:
            # If parsing fails, return empty options
            pass

        return options

    def _get_padding_name(self, padding_code: int) -> str:
        """Get padding name from code."""
        padding_map = {0: "SAME", 1: "VALID"}
        return padding_map.get(padding_code, "UNKNOWN")

    def _get_activation_name(self, activation_code: int) -> str:
        """Get activation function name from code."""
        activation_map = {
            0: "NONE",
            1: "RELU",
            2: "RELU_N1_TO_1",
            3: "RELU6",
            4: "TANH",
            5: "SIGN_BIT",
        }
        return activation_map.get(activation_code, "NONE")

    def get_tensor_by_index(self, subgraph_idx: int, tensor_idx: int) -> TensorInfo:
        """Get a tensor by its index in a specific subgraph."""
        if subgraph_idx >= len(self.subgraphs):
            raise IndexError(f"Subgraph index {subgraph_idx} out of range")

        subgraph = self.subgraphs[subgraph_idx]
        if tensor_idx >= len(subgraph.tensors):
            raise IndexError(f"Tensor index {tensor_idx} out of range")

        return subgraph.tensors[tensor_idx]

    def get_input_tensors(self, subgraph_idx: int = 0) -> list[TensorInfo]:
        """Get the input tensors for a subgraph."""
        if subgraph_idx >= len(self.subgraphs):
            raise IndexError(f"Subgraph index {subgraph_idx} out of range")

        subgraph = self.subgraphs[subgraph_idx]
        return [subgraph.tensors[idx] for idx in subgraph.inputs]

    def get_output_tensors(self, subgraph_idx: int = 0) -> list[TensorInfo]:
        """Get the output tensors for a subgraph."""
        if subgraph_idx >= len(self.subgraphs):
            raise IndexError(f"Subgraph index {subgraph_idx} out of range")

        subgraph = self.subgraphs[subgraph_idx]
        return [subgraph.tensors[idx] for idx in subgraph.outputs]

    def _extract_weights(self, subgraph, model) -> dict[int, np.ndarray]:
        """
        Extract weight tensors from the model buffers.

        Args:
            subgraph: TFLite subgraph object
            model: TFLite model object

        Returns:
            Dictionary mapping tensor index to weight data as numpy array
        """
        weights = {}

        for tensor_idx in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()

            # Skip if buffer is 0 (no data) or tensor is an input/output
            if buffer_idx == 0:
                continue

            # Get the buffer
            buffer = model.Buffers(buffer_idx)
            if buffer is None or buffer.DataLength() == 0:
                continue

            # Extract data from buffer
            data = buffer.DataAsNumpy()

            # Get tensor properties
            shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]
            dtype_code = tensor.Type()

            # Map TFLite dtype to numpy dtype
            dtype_map = {
                0: np.float32,
                1: np.float16,
                2: np.int32,
                3: np.uint8,
                4: np.int64,
                6: np.bool_,
                7: np.int16,
                8: np.complex64,
                9: np.int8,
                10: np.float64,
                11: np.complex128,
            }

            numpy_dtype = dtype_map.get(dtype_code, np.float32)

            # Reshape data to tensor shape
            try:
                weight_tensor = np.frombuffer(data, dtype=numpy_dtype).reshape(shape)
                weights[tensor_idx] = weight_tensor.copy()
            except Exception as e:
                # Skip tensors that can't be reshaped
                print(f"Warning: Could not extract weight for tensor {tensor_idx}: {e}")
                continue

        return weights
