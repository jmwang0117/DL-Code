import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test():
    torch.manual_seed(0)
    x = torch.randn(4, 3, 64, 64)

    depthwise_separable_conv = DepthwiseSeparableConv(3, 16, kernel_size=3, padding=1, stride=1)
    standard_conv = StandardConv(3, 16, kernel_size=3, padding=1, stride=1)

    out_depthwise_separable = depthwise_separable_conv(x)
    out_standard = standard_conv(x)

    print("Depthwise Separable Conv Output shape:", out_depthwise_separable.shape)
    print("Standard Conv Output shape:", out_standard.shape)

    # 计算并打印参数量
    params_depthwise_separable = count_parameters(depthwise_separable_conv)
    params_standard = count_parameters(standard_conv)

    print("Depthwise Separable Conv Parameters:", params_depthwise_separable)
    print("Standard Conv Parameters:", params_standard)


if __name__ == '__main__':
    test()