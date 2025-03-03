import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS

class EFC(BaseModule):
    def __init__(self,
                 c1, c2
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(c2)
        self.sigomid = nn.Sigmoid()
        self.group_num = 16
        self.eps = 1e-10
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, c2, 1, 1),
            nn.ReLU(True),
            nn.Softmax(dim=1),
        )
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.one = c2
        self.two = c2
        self.conv4_gobal = nn.Conv2d(c2, 1, kernel_size=1, stride=1)
        for group_id in range(0, 4):
            self.interact = nn.Conv2d(c2 // 4, c2 // 4, 1, 1, )

    def forward(self, x):
        x1, x2 = x
        global_conv1 = self.conv1(x1)
        bn_x = self.bn(global_conv1)
        weight_1 = self.sigomid(bn_x)
        global_conv2 = self.conv2(x2)
        bn_x2 = self.bn(global_conv2)
        weight_2 = self.sigomid(bn_x2)
        X_GOBAL = global_conv1 + global_conv2
        x_conv4 = self.conv4_gobal(X_GOBAL)
        X_4_sigmoid = self.sigomid(x_conv4)
        X_ = X_4_sigmoid * X_GOBAL
        X_ = X_.chunk(4, dim=1)
        out = []
        for group_id in range(0, 4):
            out_1 = self.interact(X_[group_id])
            N, C, H, W = out_1.size()
            x_1_map = out_1.reshape(N, 1, -1)
            mean_1 = x_1_map.mean(dim=2, keepdim=True)
            x_1_av = x_1_map / mean_1
            x_2_2 = F.softmax(x_1_av, dim=-1)
            x1 = x_2_2.reshape(N, C, H, W)
            x1 = X_[group_id] * x1
            out.append(x1)
        out = torch.cat([out[0], out[1], out[2], out[3]], dim=1)
        N, C, H, W = out.size()
        x_add_1 = out.reshape(N, self.group_num, -1)
        N, C, H, W = X_GOBAL.size()
        x_shape_1 = X_GOBAL.reshape(N, self.group_num, -1)
        mean_1 = x_shape_1.mean(dim=2, keepdim=True)
        std_1 = x_shape_1.std(dim=2, keepdim=True)
        x_guiyi = (x_add_1 - mean_1) / (std_1 + self.eps)
        x_guiyi_1 = x_guiyi.reshape(N, C, H, W)
        x_gui = (x_guiyi_1 * self.gamma + self.beta)
        weight_x3 = self.Apt(X_GOBAL)
        reweights = self.sigomid(weight_x3)
        x_up_1 = reweights >= weight_1
        x_low_1 = reweights < weight_1
        x_up_2 = reweights >= weight_2
        x_low_2 = reweights < weight_2
        x_up = x_up_1 * X_GOBAL + x_up_2 * X_GOBAL
        x_low = x_low_1 * X_GOBAL + x_low_2 * X_GOBAL
        x11_up_dwc = self.dwconv(x_low)
        x11_up_dwc = self.conv3(x11_up_dwc)
        x_so = self.gate_genator(x_low)
        x11_up_dwc = x11_up_dwc * x_so
        x22_low_pw = self.conv4(x_up)
        xL = x11_up_dwc + x22_low_pw
        xL = xL + x_gui

        return xL


class Maxpoll(BaseModule):  # 串联
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=3 // 2)
        self.conv3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=3 // 2)
        self.conv2 = nn.Conv2d(2 * dim, dim_out, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.conv2(x3)
        return x4


@NECKS.register_module()
class FPN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super(FPN, self).__init__()
        self.P5_1 = nn.Conv2d(in_channels[3], out_channels, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.efc1 = EFC(in_channels[2], out_channels)
        self.efc2 = EFC(in_channels[1], out_channels)
        self.efc3 = EFC(out_channels, out_channels)
        self.efc4 = EFC(out_channels, out_channels)
        self.P6 = Maxpoll(out_channels, out_channels)
        self.P7 = Maxpoll(out_channels, out_channels)
        self.down_one = Maxpoll(out_channels, out_channels)
        self.down_two = Maxpoll(out_channels, out_channels)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""

        C2, C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)  # 512-256  10
        P5_x = self.P5_2(P5_x)  # 256-256  10
        P5_upsampled_x = self.P5_upsampled(P5_x)  # 256  20
        P4_x = self.efc1([C4, P5_upsampled_x])  # 256  20
        P4_upsampled_x = self.P4_upsampled(P4_x)  # 256  40
        P3_x = self.efc2([C3, P4_upsampled_x])
        P_down1 = self.down_one(P3_x)
        P_4 = self.efc3([P4_x, P_down1])
        P_down2 = self.down_two(P_4)
        P_5 = self.efc4([P5_x, P_down2])
        P6_x = self.P6(P_5)
        P7_x = self.P7(P6_x)

        return [P3_x, P_4, P_5, P6_x, P7_x]
