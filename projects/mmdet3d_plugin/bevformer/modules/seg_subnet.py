import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18
from ..modules.builder import SEG_ENCODER
from .edsr import make_edsr_baseline

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension

class BevFeatureSlicer(nn.Module):
    # crop the interested area in BEV feature for semantic map segmentation
    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()

        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False

            bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
                grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
            )

            map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound'],
            )

            self.map_x = torch.arange(
                map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])

            self.map_y = torch.arange(
                map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

            # convert to normalized coords
            self.norm_map_x = self.map_x / (- bev_start_position[0])
            self.norm_map_y = self.map_y / (- bev_start_position[1])
            # vision 1 失败
            self.map_grid = torch.stack(torch.meshgrid(
                self.norm_map_x, self.norm_map_y), dim=2).permute(1, 0, 2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)

            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1) #相当与通道维度上连接，以弥补因为使用mb导致的卷积信息丢失。
        return self.conv(x1)

@SEG_ENCODER.register_module()
class SegEncode(nn.Module):
    def __init__(self, inC, outC, det_grid_conf, map_grid_conf, tnn=False):
        super(SegEncode, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.tnn = tnn
        self.feat_cropper = BevFeatureSlicer(det_grid_conf, map_grid_conf)
        if self.tnn:
            self.seg_sr = make_edsr_baseline(in_ch=128, n_feats=512, out_ch=128, scale=4, no_upsampling=False)
        self.seg_decoder = nn.Conv2d(128, outC, kernel_size=1, padding=0)

    def forward(self, x): #torch.Size([2, 256, 200, 400])
        x = self.conv1(x) #torch.Size([2, 64, 200, 400])
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) #torch.Size([2, 64, 100, 200])
        x = self.layer2(x1) #torch.Size([2, 128, 50, 100])
        x2 = self.layer3(x) #torch.Size([2, 256, 25, 50])

        x = self.up1(x2, x1) #torch.Size([2, 256, 100, 200])
        x = self.up2(x) #torch.Size([2, 4, 200, 400]) 语义分割预测特征图

        x = self.feat_cropper(x)
        if self.tnn:
            x = self.seg_sr(x)
        x = self.seg_decoder(x)

        return x

@SEG_ENCODER.register_module()
class SegEncode_v1(nn.Module):

    def __init__(self, inC, outC):
        super(SegEncode_v1, self).__init__()
        self.seg_head = nn.Sequential(
            nn.Conv2d(inC, inC, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC , kernel_size=1))

    def forward(self, x):

        return self.seg_head(x)


import math
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16


@SEG_ENCODER.register_module()
class DeconvEncode(BaseModule):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.

    Args:
         in_channel (int): Number of input channels.
         num_deconv_filters (tuple[int]): Number of filters per stage.
         num_deconv_kernels (tuple[int]): Number of kernels per stage.
         use_dcn (bool): If True, use DCNv2. Default: True.
         init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channel,
                 num_deconv_filters,
                 num_deconv_kernels,
                 outC=4,
                 use_dcn=True,
                 init_cfg=None):
        super(DeconvEncode, self).__init__(init_cfg)
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channel = in_channel
        self.deconv_layers = self._make_deconv_layer(num_deconv_filters,
                                                     num_deconv_kernels)

        self.seg_head = nn.Sequential(
            nn.Conv2d(num_deconv_filters[-1], num_deconv_filters[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_deconv_filters[-1], outC , kernel_size=1))

    def _make_deconv_layer(self, num_deconv_filters, num_deconv_kernels):
        """use deconv layers to upsample backbone's output."""
        layers = []
        for i in range(len(num_deconv_filters)):
            feat_channel = num_deconv_filters[i]
            conv_module = ConvModule(
                self.in_channel,
                feat_channel,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=dict(type='BN'))
            layers.append(conv_module)
            upsample_module = ConvModule(
                feat_channel,
                feat_channel,
                num_deconv_kernels[i],
                stride=2,
                padding=1,
                conv_cfg=dict(type='deconv'),
                norm_cfg=dict(type='BN'))
            layers.append(upsample_module)
            self.in_channel = feat_channel

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    @auto_fp16()
    def forward(self, inputs):
        outs = self.deconv_layers(inputs)
        outs = self.seg_head(outs)
        return outs

"""

dict(
type='DeconvEncode',
in_channel=256,
num_deconv_filters=(256, 128, 64),
num_deconv_kernels=(4, 4, 4),
use_dcn=True),
        
"""
