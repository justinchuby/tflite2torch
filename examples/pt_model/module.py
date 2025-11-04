
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class FxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = Linear(in_features=257, out_features=64, bias=True)
        self.module_1 = Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2))
        self.activation_2 = torch.load(r'examples/pt_model/activation_2.pt', weights_only=False) # ReLU6()
        self.module_3 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=32)
        self.module_4 = Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_5 = torch.load(r'examples/pt_model/activation_5.pt', weights_only=False) # ReLU6()
        self.module_6 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=64)
        self.module_7 = Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_8 = torch.load(r'examples/pt_model/activation_8.pt', weights_only=False) # ReLU6()
        self.module_9 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=128)
        self.module_10 = Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_11 = torch.load(r'examples/pt_model/activation_11.pt', weights_only=False) # ReLU6()
        self.module_12 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=128)
        self.module_13 = Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_14 = torch.load(r'examples/pt_model/activation_14.pt', weights_only=False) # ReLU6()
        self.module_15 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=256)
        self.module_16 = Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_17 = torch.load(r'examples/pt_model/activation_17.pt', weights_only=False) # ReLU6()
        self.module_18 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=256)
        self.module_19 = Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_20 = torch.load(r'examples/pt_model/activation_20.pt', weights_only=False) # ReLU6()
        self.module_21 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=512)
        self.module_22 = Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_23 = torch.load(r'examples/pt_model/activation_23.pt', weights_only=False) # ReLU6()
        self.module_24 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=512)
        self.module_25 = Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_26 = torch.load(r'examples/pt_model/activation_26.pt', weights_only=False) # ReLU6()
        self.module_27 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=512)
        self.module_28 = Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_29 = torch.load(r'examples/pt_model/activation_29.pt', weights_only=False) # ReLU6()
        self.module_30 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=512)
        self.module_31 = Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_32 = torch.load(r'examples/pt_model/activation_32.pt', weights_only=False) # ReLU6()
        self.module_33 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=512)
        self.module_34 = Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_35 = torch.load(r'examples/pt_model/activation_35.pt', weights_only=False) # ReLU6()
        self.module_36 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=512)
        self.module_37 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_38 = torch.load(r'examples/pt_model/activation_38.pt', weights_only=False) # ReLU6()
        self.module_39 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=same, groups=1024)
        self.module_40 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), padding=same)
        self.activation_41 = torch.load(r'examples/pt_model/activation_41.pt', weights_only=False) # ReLU6()
        self.module_42 = Linear(in_features=1024, out_features=521, bias=True)
        self.module_43 = torch.load(r'examples/pt_model/module_43.pt', weights_only=False) # Sigmoid()
        self.stft_frame_zeros_like = torch.nn.Parameter(torch.empty([1], dtype=torch.float32))
        self.stft_frame_concat = torch.nn.Parameter(torch.empty([1], dtype=torch.float32))
        self.stft_frame_ones_like = torch.nn.Parameter(torch.empty([1], dtype=torch.float32))
        self.stft_frame_concat_1 = torch.nn.Parameter(torch.empty([2], dtype=torch.float32))
        self.stft_frame_add_1 = torch.nn.Parameter(torch.empty([96, 5], dtype=torch.float32))
        self.stft_frame_concat_2_values_1 = torch.nn.Parameter(torch.empty([2], dtype=torch.float32))
        self.stft_hann_window_sub_2 = torch.nn.Parameter(torch.empty([400], dtype=torch.float32))
        self.stft_rfft_Pad_paddings = torch.nn.Parameter(torch.empty([2, 2], dtype=torch.float32))
        self.stft_rfft1 = torch.nn.Parameter(torch.empty([3], dtype=torch.float32))
        self.stft_rfft = torch.nn.Parameter(torch.empty([2], dtype=torch.float32))
        self.stft_rfft2 = torch.nn.Parameter(torch.empty([2], dtype=torch.float32))
        self.mel_spectrogram = torch.nn.Parameter(torch.empty([64, 257], dtype=torch.float32))
        self.add = torch.nn.Parameter(torch.empty([64], dtype=torch.float32))
        self.Reshape_shape = torch.nn.Parameter(torch.empty([3], dtype=torch.float32))
        self.ExpandDims = torch.nn.Parameter(torch.empty([4], dtype=torch.float32))
        self.pre_tower_split_split_dim = torch.nn.Parameter(torch.empty([], dtype=torch.float32))
        self.tower0_network_layer1_conv_Conv2D = torch.nn.Parameter(torch.empty([32, 3, 3, 1], dtype=torch.float32))
        self.tower0_network_layer1_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([32], dtype=torch.float32))
        self.tower0_network_layer2_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer2_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 32], dtype=torch.float32))
        self.tower0_network_layer2_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([32], dtype=torch.float32))
        self.tower0_network_layer3_conv_Conv2D = torch.nn.Parameter(torch.empty([64, 1, 1, 32], dtype=torch.float32))
        self.tower0_network_layer3_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([64], dtype=torch.float32))
        self.tower0_network_layer4_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer4_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 64], dtype=torch.float32))
        self.tower0_network_layer4_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([64], dtype=torch.float32))
        self.tower0_network_layer5_conv_Conv2D = torch.nn.Parameter(torch.empty([128, 1, 1, 64], dtype=torch.float32))
        self.tower0_network_layer5_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([128], dtype=torch.float32))
        self.tower0_network_layer6_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer6_sepconv_depthwise;tower0_network_layer8_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 128], dtype=torch.float32))
        self.tower0_network_layer6_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([128], dtype=torch.float32))
        self.tower0_network_layer7_conv_Conv2D = torch.nn.Parameter(torch.empty([128, 1, 1, 128], dtype=torch.float32))
        self.tower0_network_layer7_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([128], dtype=torch.float32))
        self.tower0_network_layer8_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer8_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 128], dtype=torch.float32))
        self.tower0_network_layer8_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([128], dtype=torch.float32))
        self.tower0_network_layer9_conv_Conv2D = torch.nn.Parameter(torch.empty([256, 1, 1, 128], dtype=torch.float32))
        self.tower0_network_layer9_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([256], dtype=torch.float32))
        self.tower0_network_layer10_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer10_sepconv_depthwise;tower0_network_layer12_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 256], dtype=torch.float32))
        self.tower0_network_layer10_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([256], dtype=torch.float32))
        self.tower0_network_layer11_conv_Conv2D = torch.nn.Parameter(torch.empty([256, 1, 1, 256], dtype=torch.float32))
        self.tower0_network_layer11_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([256], dtype=torch.float32))
        self.tower0_network_layer12_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer12_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 256], dtype=torch.float32))
        self.tower0_network_layer12_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([256], dtype=torch.float32))
        self.tower0_network_layer13_conv_Conv2D = torch.nn.Parameter(torch.empty([512, 1, 1, 256], dtype=torch.float32))
        self.tower0_network_layer13_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer14_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer14_sepconv_depthwise;tower0_network_layer24_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 512], dtype=torch.float32))
        self.tower0_network_layer14_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer15_conv_Conv2D = torch.nn.Parameter(torch.empty([512, 1, 1, 512], dtype=torch.float32))
        self.tower0_network_layer15_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer16_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer16_sepconv_depthwise;tower0_network_layer24_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 512], dtype=torch.float32))
        self.tower0_network_layer16_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer17_conv_Conv2D = torch.nn.Parameter(torch.empty([512, 1, 1, 512], dtype=torch.float32))
        self.tower0_network_layer17_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer18_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer18_sepconv_depthwise;tower0_network_layer24_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 512], dtype=torch.float32))
        self.tower0_network_layer18_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer19_conv_Conv2D = torch.nn.Parameter(torch.empty([512, 1, 1, 512], dtype=torch.float32))
        self.tower0_network_layer19_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer20_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer20_sepconv_depthwise;tower0_network_layer24_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 512], dtype=torch.float32))
        self.tower0_network_layer20_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer21_conv_Conv2D = torch.nn.Parameter(torch.empty([512, 1, 1, 512], dtype=torch.float32))
        self.tower0_network_layer21_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer22_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer22_sepconv_depthwise;tower0_network_layer24_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 512], dtype=torch.float32))
        self.tower0_network_layer22_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer23_conv_Conv2D = torch.nn.Parameter(torch.empty([512, 1, 1, 512], dtype=torch.float32))
        self.tower0_network_layer23_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer24_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer24_sepconv_depthwise = torch.nn.Parameter(torch.empty([1, 3, 3, 512], dtype=torch.float32))
        self.tower0_network_layer24_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([512], dtype=torch.float32))
        self.tower0_network_layer25_conv_Conv2D = torch.nn.Parameter(torch.empty([1024, 1, 1, 512], dtype=torch.float32))
        self.tower0_network_layer25_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([1024], dtype=torch.float32))
        self.tower0_network_layer26_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer26_sepconv_depthwise;tower0_network_layer27_conv_Conv2D = torch.nn.Parameter(torch.empty([1, 3, 3, 1024], dtype=torch.float32))
        self.tower0_network_layer26_sepconv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([1024], dtype=torch.float32))
        self.tower0_network_layer27_conv_Conv2D = torch.nn.Parameter(torch.empty([1024, 1, 1, 1024], dtype=torch.float32))
        self.tower0_network_layer27_conv_BatchNorm_FusedBatchNormV3 = torch.nn.Parameter(torch.empty([1024], dtype=torch.float32))
        self.tower0_network_layer28_reduce_mean_reduction_indices = torch.nn.Parameter(torch.empty([2], dtype=torch.float32))
        self.tower0_network_layer29_fc_MatMul = torch.nn.Parameter(torch.empty([521, 1024], dtype=torch.float32))
        self.network_layer29_fc_biases = torch.nn.Parameter(torch.empty([521], dtype=torch.float32))
        self.load_state_dict(torch.load(r'examples/pt_model/state_dict.pt'))

    
    
    def forward(self, waveform_binary):
        stft_frame_zeros_like = self.stft_frame_zeros_like
        stft_frame_concat = self.stft_frame_concat
        stft_frame_ones_like = self.stft_frame_ones_like
        strided_slice_0 = tflite2torch__operator_converter_strided_slice_impl(waveform_binary, stft_frame_zeros_like, stft_frame_concat, stft_frame_ones_like);  waveform_binary = stft_frame_zeros_like = stft_frame_concat = stft_frame_ones_like = None
        stft_frame_concat_1 = self.stft_frame_concat_1;  stft_frame_concat_1 = None
        reshape_1 = torch.reshape(strided_slice_0, (195.0, 80.0));  strided_slice_0 = None
        stft_frame_add_1 = self.stft_frame_add_1
        gather_2 = torch.index_select(reshape_1, 0, stft_frame_add_1);  reshape_1 = stft_frame_add_1 = None
        stft_frame_concat_2_values_1 = self.stft_frame_concat_2_values_1;  stft_frame_concat_2_values_1 = None
        reshape_3 = torch.reshape(gather_2, (96.0, 400.0));  gather_2 = None
        stft_hann_window_sub_2 = self.stft_hann_window_sub_2
        mul_4 = torch.mul(reshape_3, stft_hann_window_sub_2);  reshape_3 = stft_hann_window_sub_2 = None
        stft_rfft_pad_paddings = self.stft_rfft_Pad_paddings
        pad_5 = tflite2torch__operator_converter_convert_pad(mul_4, stft_rfft_pad_paddings);  mul_4 = stft_rfft_pad_paddings = None
        stft_rfft1 = self.stft_rfft1;  stft_rfft1 = None
        reshape_6 = torch.reshape(pad_5, (96.0, 1.0, 512.0));  pad_5 = None
        stft_rfft = self.stft_rfft
        rfft2d_7 = tflite2torch__operator_converter_rfft2d_with_length(reshape_6, stft_rfft);  reshape_6 = stft_rfft = None
        stft_rfft2 = self.stft_rfft2;  stft_rfft2 = None
        reshape_8 = torch.reshape(rfft2d_7, (96.0, 257.0));  rfft2d_7 = None
        unsupported_complex_abs_9 = tflite2torch__fx_reconstructor_lambda_args_args_0_if_args_else_None_(reshape_8);  reshape_8 = None
        mel_spectrogram = self.mel_spectrogram;  mel_spectrogram = None
        add = self.add;  add = None
        fully_connected_10 = self.module_0(unsupported_complex_abs_9);  unsupported_complex_abs_9 = None
        log_11 = torch.log(fully_connected_10);  fully_connected_10 = None
        reshape_shape = self.Reshape_shape;  reshape_shape = None
        reshape_12 = torch.reshape(log_11, (1.0, 96.0, 64.0));  log_11 = None
        quantize_13 = torch.quantize_per_tensor(reshape_12);  reshape_12 = None
        expand_dims = self.ExpandDims;  expand_dims = None
        reshape_14 = torch.reshape(quantize_13, (1.0, 96.0, 64.0, 1.0));  quantize_13 = None
        pre_tower_split_split_dim = self.pre_tower_split_split_dim
        split_15 = tflite2torch__operator_converter_split_impl(reshape_14, pre_tower_split_split_dim, 1);  reshape_14 = pre_tower_split_split_dim = None
        tower0_network_layer1_conv_conv2d = self.tower0_network_layer1_conv_Conv2D;  tower0_network_layer1_conv_conv2d = None
        tower0_network_layer1_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer1_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer1_conv_batch_norm_fused_batch_norm_v3 = None
        permute = torch.permute(split_15, (0, 3, 1, 2));  split_15 = None
        calc_same_padding = tflite2torch__operator_converter_calc_same_padding(permute, 3, 3, 2, 2, 1, 1);  permute = None
        module_1 = self.module_1(calc_same_padding);  calc_same_padding = None
        conv_2d_16 = torch.permute(module_1, (0, 2, 3, 1));  module_1 = None
        conv_2d_16_activation = self.activation_2(conv_2d_16);  conv_2d_16 = None
        tower0_network_layer2_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer2_sepconv_depthwise = getattr(self, "tower0_network_layer2_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer2_sepconv_depthwise");  tower0_network_layer2_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer2_sepconv_depthwise = None
        tower0_network_layer2_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer2_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer2_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_2 = torch.permute(conv_2d_16_activation, (0, 3, 1, 2));  conv_2d_16_activation = None
        module_3 = self.module_3(permute_2);  permute_2 = None
        depthwise_conv_2d_17 = torch.permute(module_3, (0, 2, 3, 1));  module_3 = None
        tower0_network_layer3_conv_conv2d = self.tower0_network_layer3_conv_Conv2D;  tower0_network_layer3_conv_conv2d = None
        tower0_network_layer3_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer3_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer3_conv_batch_norm_fused_batch_norm_v3 = None
        permute_4 = torch.permute(depthwise_conv_2d_17, (0, 3, 1, 2));  depthwise_conv_2d_17 = None
        module_4 = self.module_4(permute_4);  permute_4 = None
        conv_2d_18 = torch.permute(module_4, (0, 2, 3, 1));  module_4 = None
        conv_2d_18_activation = self.activation_5(conv_2d_18);  conv_2d_18 = None
        tower0_network_layer4_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer4_sepconv_depthwise = getattr(self, "tower0_network_layer4_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer4_sepconv_depthwise");  tower0_network_layer4_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer4_sepconv_depthwise = None
        tower0_network_layer4_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer4_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer4_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_6 = torch.permute(conv_2d_18_activation, (0, 3, 1, 2));  conv_2d_18_activation = None
        module_6 = self.module_6(permute_6);  permute_6 = None
        depthwise_conv_2d_19 = torch.permute(module_6, (0, 2, 3, 1));  module_6 = None
        tower0_network_layer5_conv_conv2d = self.tower0_network_layer5_conv_Conv2D;  tower0_network_layer5_conv_conv2d = None
        tower0_network_layer5_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer5_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer5_conv_batch_norm_fused_batch_norm_v3 = None
        permute_8 = torch.permute(depthwise_conv_2d_19, (0, 3, 1, 2));  depthwise_conv_2d_19 = None
        module_7 = self.module_7(permute_8);  permute_8 = None
        conv_2d_20 = torch.permute(module_7, (0, 2, 3, 1));  module_7 = None
        conv_2d_20_activation = self.activation_8(conv_2d_20);  conv_2d_20 = None
        tower0_network_layer6_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer6_sepconv_depthwise_tower0_network_layer8_sepconv_depthwise = getattr(self, "tower0_network_layer6_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer6_sepconv_depthwise;tower0_network_layer8_sepconv_depthwise");  tower0_network_layer6_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer6_sepconv_depthwise_tower0_network_layer8_sepconv_depthwise = None
        tower0_network_layer6_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer6_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer6_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_10 = torch.permute(conv_2d_20_activation, (0, 3, 1, 2));  conv_2d_20_activation = None
        module_9 = self.module_9(permute_10);  permute_10 = None
        depthwise_conv_2d_21 = torch.permute(module_9, (0, 2, 3, 1));  module_9 = None
        tower0_network_layer7_conv_conv2d = self.tower0_network_layer7_conv_Conv2D;  tower0_network_layer7_conv_conv2d = None
        tower0_network_layer7_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer7_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer7_conv_batch_norm_fused_batch_norm_v3 = None
        permute_12 = torch.permute(depthwise_conv_2d_21, (0, 3, 1, 2));  depthwise_conv_2d_21 = None
        module_10 = self.module_10(permute_12);  permute_12 = None
        conv_2d_22 = torch.permute(module_10, (0, 2, 3, 1));  module_10 = None
        conv_2d_22_activation = self.activation_11(conv_2d_22);  conv_2d_22 = None
        tower0_network_layer8_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer8_sepconv_depthwise = getattr(self, "tower0_network_layer8_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer8_sepconv_depthwise");  tower0_network_layer8_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer8_sepconv_depthwise = None
        tower0_network_layer8_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer8_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer8_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_14 = torch.permute(conv_2d_22_activation, (0, 3, 1, 2));  conv_2d_22_activation = None
        module_12 = self.module_12(permute_14);  permute_14 = None
        depthwise_conv_2d_23 = torch.permute(module_12, (0, 2, 3, 1));  module_12 = None
        tower0_network_layer9_conv_conv2d = self.tower0_network_layer9_conv_Conv2D;  tower0_network_layer9_conv_conv2d = None
        tower0_network_layer9_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer9_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer9_conv_batch_norm_fused_batch_norm_v3 = None
        permute_16 = torch.permute(depthwise_conv_2d_23, (0, 3, 1, 2));  depthwise_conv_2d_23 = None
        module_13 = self.module_13(permute_16);  permute_16 = None
        conv_2d_24 = torch.permute(module_13, (0, 2, 3, 1));  module_13 = None
        conv_2d_24_activation = self.activation_14(conv_2d_24);  conv_2d_24 = None
        tower0_network_layer10_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer10_sepconv_depthwise_tower0_network_layer12_sepconv_depthwise = getattr(self, "tower0_network_layer10_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer10_sepconv_depthwise;tower0_network_layer12_sepconv_depthwise");  tower0_network_layer10_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer10_sepconv_depthwise_tower0_network_layer12_sepconv_depthwise = None
        tower0_network_layer10_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer10_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer10_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_18 = torch.permute(conv_2d_24_activation, (0, 3, 1, 2));  conv_2d_24_activation = None
        module_15 = self.module_15(permute_18);  permute_18 = None
        depthwise_conv_2d_25 = torch.permute(module_15, (0, 2, 3, 1));  module_15 = None
        tower0_network_layer11_conv_conv2d = self.tower0_network_layer11_conv_Conv2D;  tower0_network_layer11_conv_conv2d = None
        tower0_network_layer11_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer11_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer11_conv_batch_norm_fused_batch_norm_v3 = None
        permute_20 = torch.permute(depthwise_conv_2d_25, (0, 3, 1, 2));  depthwise_conv_2d_25 = None
        module_16 = self.module_16(permute_20);  permute_20 = None
        conv_2d_26 = torch.permute(module_16, (0, 2, 3, 1));  module_16 = None
        conv_2d_26_activation = self.activation_17(conv_2d_26);  conv_2d_26 = None
        tower0_network_layer12_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer12_sepconv_depthwise = getattr(self, "tower0_network_layer12_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer12_sepconv_depthwise");  tower0_network_layer12_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer12_sepconv_depthwise = None
        tower0_network_layer12_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer12_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer12_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_22 = torch.permute(conv_2d_26_activation, (0, 3, 1, 2));  conv_2d_26_activation = None
        module_18 = self.module_18(permute_22);  permute_22 = None
        depthwise_conv_2d_27 = torch.permute(module_18, (0, 2, 3, 1));  module_18 = None
        tower0_network_layer13_conv_conv2d = self.tower0_network_layer13_conv_Conv2D;  tower0_network_layer13_conv_conv2d = None
        tower0_network_layer13_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer13_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer13_conv_batch_norm_fused_batch_norm_v3 = None
        permute_24 = torch.permute(depthwise_conv_2d_27, (0, 3, 1, 2));  depthwise_conv_2d_27 = None
        module_19 = self.module_19(permute_24);  permute_24 = None
        conv_2d_28 = torch.permute(module_19, (0, 2, 3, 1));  module_19 = None
        conv_2d_28_activation = self.activation_20(conv_2d_28);  conv_2d_28 = None
        tower0_network_layer14_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer14_sepconv_depthwise_tower0_network_layer24_sepconv_depthwise = getattr(self, "tower0_network_layer14_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer14_sepconv_depthwise;tower0_network_layer24_sepconv_depthwise");  tower0_network_layer14_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer14_sepconv_depthwise_tower0_network_layer24_sepconv_depthwise = None
        tower0_network_layer14_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer14_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer14_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_26 = torch.permute(conv_2d_28_activation, (0, 3, 1, 2));  conv_2d_28_activation = None
        module_21 = self.module_21(permute_26);  permute_26 = None
        depthwise_conv_2d_29 = torch.permute(module_21, (0, 2, 3, 1));  module_21 = None
        tower0_network_layer15_conv_conv2d = self.tower0_network_layer15_conv_Conv2D;  tower0_network_layer15_conv_conv2d = None
        tower0_network_layer15_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer15_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer15_conv_batch_norm_fused_batch_norm_v3 = None
        permute_28 = torch.permute(depthwise_conv_2d_29, (0, 3, 1, 2));  depthwise_conv_2d_29 = None
        module_22 = self.module_22(permute_28);  permute_28 = None
        conv_2d_30 = torch.permute(module_22, (0, 2, 3, 1));  module_22 = None
        conv_2d_30_activation = self.activation_23(conv_2d_30);  conv_2d_30 = None
        tower0_network_layer16_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer16_sepconv_depthwise_tower0_network_layer24_sepconv_depthwise = getattr(self, "tower0_network_layer16_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer16_sepconv_depthwise;tower0_network_layer24_sepconv_depthwise");  tower0_network_layer16_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer16_sepconv_depthwise_tower0_network_layer24_sepconv_depthwise = None
        tower0_network_layer16_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer16_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer16_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_30 = torch.permute(conv_2d_30_activation, (0, 3, 1, 2));  conv_2d_30_activation = None
        module_24 = self.module_24(permute_30);  permute_30 = None
        depthwise_conv_2d_31 = torch.permute(module_24, (0, 2, 3, 1));  module_24 = None
        tower0_network_layer17_conv_conv2d = self.tower0_network_layer17_conv_Conv2D;  tower0_network_layer17_conv_conv2d = None
        tower0_network_layer17_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer17_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer17_conv_batch_norm_fused_batch_norm_v3 = None
        permute_32 = torch.permute(depthwise_conv_2d_31, (0, 3, 1, 2));  depthwise_conv_2d_31 = None
        module_25 = self.module_25(permute_32);  permute_32 = None
        conv_2d_32 = torch.permute(module_25, (0, 2, 3, 1));  module_25 = None
        conv_2d_32_activation = self.activation_26(conv_2d_32);  conv_2d_32 = None
        tower0_network_layer18_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer18_sepconv_depthwise_tower0_network_layer24_sepconv_depthwise = getattr(self, "tower0_network_layer18_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer18_sepconv_depthwise;tower0_network_layer24_sepconv_depthwise");  tower0_network_layer18_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer18_sepconv_depthwise_tower0_network_layer24_sepconv_depthwise = None
        tower0_network_layer18_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer18_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer18_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_34 = torch.permute(conv_2d_32_activation, (0, 3, 1, 2));  conv_2d_32_activation = None
        module_27 = self.module_27(permute_34);  permute_34 = None
        depthwise_conv_2d_33 = torch.permute(module_27, (0, 2, 3, 1));  module_27 = None
        tower0_network_layer19_conv_conv2d = self.tower0_network_layer19_conv_Conv2D;  tower0_network_layer19_conv_conv2d = None
        tower0_network_layer19_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer19_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer19_conv_batch_norm_fused_batch_norm_v3 = None
        permute_36 = torch.permute(depthwise_conv_2d_33, (0, 3, 1, 2));  depthwise_conv_2d_33 = None
        module_28 = self.module_28(permute_36);  permute_36 = None
        conv_2d_34 = torch.permute(module_28, (0, 2, 3, 1));  module_28 = None
        conv_2d_34_activation = self.activation_29(conv_2d_34);  conv_2d_34 = None
        tower0_network_layer20_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer20_sepconv_depthwise_tower0_network_layer24_sepconv_depthwise = getattr(self, "tower0_network_layer20_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer20_sepconv_depthwise;tower0_network_layer24_sepconv_depthwise");  tower0_network_layer20_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer20_sepconv_depthwise_tower0_network_layer24_sepconv_depthwise = None
        tower0_network_layer20_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer20_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer20_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_38 = torch.permute(conv_2d_34_activation, (0, 3, 1, 2));  conv_2d_34_activation = None
        module_30 = self.module_30(permute_38);  permute_38 = None
        depthwise_conv_2d_35 = torch.permute(module_30, (0, 2, 3, 1));  module_30 = None
        tower0_network_layer21_conv_conv2d = self.tower0_network_layer21_conv_Conv2D;  tower0_network_layer21_conv_conv2d = None
        tower0_network_layer21_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer21_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer21_conv_batch_norm_fused_batch_norm_v3 = None
        permute_40 = torch.permute(depthwise_conv_2d_35, (0, 3, 1, 2));  depthwise_conv_2d_35 = None
        module_31 = self.module_31(permute_40);  permute_40 = None
        conv_2d_36 = torch.permute(module_31, (0, 2, 3, 1));  module_31 = None
        conv_2d_36_activation = self.activation_32(conv_2d_36);  conv_2d_36 = None
        tower0_network_layer22_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer22_sepconv_depthwise_tower0_network_layer24_sepconv_depthwise = getattr(self, "tower0_network_layer22_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer22_sepconv_depthwise;tower0_network_layer24_sepconv_depthwise");  tower0_network_layer22_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer22_sepconv_depthwise_tower0_network_layer24_sepconv_depthwise = None
        tower0_network_layer22_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer22_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer22_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_42 = torch.permute(conv_2d_36_activation, (0, 3, 1, 2));  conv_2d_36_activation = None
        module_33 = self.module_33(permute_42);  permute_42 = None
        depthwise_conv_2d_37 = torch.permute(module_33, (0, 2, 3, 1));  module_33 = None
        tower0_network_layer23_conv_conv2d = self.tower0_network_layer23_conv_Conv2D;  tower0_network_layer23_conv_conv2d = None
        tower0_network_layer23_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer23_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer23_conv_batch_norm_fused_batch_norm_v3 = None
        permute_44 = torch.permute(depthwise_conv_2d_37, (0, 3, 1, 2));  depthwise_conv_2d_37 = None
        module_34 = self.module_34(permute_44);  permute_44 = None
        conv_2d_38 = torch.permute(module_34, (0, 2, 3, 1));  module_34 = None
        conv_2d_38_activation = self.activation_35(conv_2d_38);  conv_2d_38 = None
        tower0_network_layer24_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer24_sepconv_depthwise = getattr(self, "tower0_network_layer24_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer24_sepconv_depthwise");  tower0_network_layer24_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer24_sepconv_depthwise = None
        tower0_network_layer24_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer24_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer24_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_46 = torch.permute(conv_2d_38_activation, (0, 3, 1, 2));  conv_2d_38_activation = None
        module_36 = self.module_36(permute_46);  permute_46 = None
        depthwise_conv_2d_39 = torch.permute(module_36, (0, 2, 3, 1));  module_36 = None
        tower0_network_layer25_conv_conv2d = self.tower0_network_layer25_conv_Conv2D;  tower0_network_layer25_conv_conv2d = None
        tower0_network_layer25_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer25_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer25_conv_batch_norm_fused_batch_norm_v3 = None
        permute_48 = torch.permute(depthwise_conv_2d_39, (0, 3, 1, 2));  depthwise_conv_2d_39 = None
        module_37 = self.module_37(permute_48);  permute_48 = None
        conv_2d_40 = torch.permute(module_37, (0, 2, 3, 1));  module_37 = None
        conv_2d_40_activation = self.activation_38(conv_2d_40);  conv_2d_40 = None
        tower0_network_layer26_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer26_sepconv_depthwise_tower0_network_layer27_conv_conv2d = getattr(self, "tower0_network_layer26_sepconv_BatchNorm_FusedBatchNormV3;tower0_network_layer26_sepconv_depthwise;tower0_network_layer27_conv_Conv2D");  tower0_network_layer26_sepconv_batch_norm_fused_batch_norm_v3_tower0_network_layer26_sepconv_depthwise_tower0_network_layer27_conv_conv2d = None
        tower0_network_layer26_sepconv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer26_sepconv_BatchNorm_FusedBatchNormV3;  tower0_network_layer26_sepconv_batch_norm_fused_batch_norm_v3 = None
        permute_50 = torch.permute(conv_2d_40_activation, (0, 3, 1, 2));  conv_2d_40_activation = None
        module_39 = self.module_39(permute_50);  permute_50 = None
        depthwise_conv_2d_41 = torch.permute(module_39, (0, 2, 3, 1));  module_39 = None
        tower0_network_layer27_conv_conv2d = self.tower0_network_layer27_conv_Conv2D;  tower0_network_layer27_conv_conv2d = None
        tower0_network_layer27_conv_batch_norm_fused_batch_norm_v3 = self.tower0_network_layer27_conv_BatchNorm_FusedBatchNormV3;  tower0_network_layer27_conv_batch_norm_fused_batch_norm_v3 = None
        permute_52 = torch.permute(depthwise_conv_2d_41, (0, 3, 1, 2));  depthwise_conv_2d_41 = None
        module_40 = self.module_40(permute_52);  permute_52 = None
        conv_2d_42 = torch.permute(module_40, (0, 2, 3, 1));  module_40 = None
        conv_2d_42_activation = self.activation_41(conv_2d_42);  conv_2d_42 = None
        tower0_network_layer28_reduce_mean_reduction_indices = self.tower0_network_layer28_reduce_mean_reduction_indices;  tower0_network_layer28_reduce_mean_reduction_indices = None
        mean_43 = torch.mean(conv_2d_42_activation, dim = (1.0, 2.0), keepdim = True);  conv_2d_42_activation = None
        tower0_network_layer29_fc_mat_mul = self.tower0_network_layer29_fc_MatMul;  tower0_network_layer29_fc_mat_mul = None
        network_layer29_fc_biases = self.network_layer29_fc_biases;  network_layer29_fc_biases = None
        fully_connected_44 = self.module_42(mean_43);  mean_43 = None
        logistic_45 = self.module_43(fully_connected_44);  fully_connected_44 = None
        dequantize_46 = tflite2torch__operator_converter_output_node(logistic_45);  logistic_45 = None
        return dequantize_46
        
