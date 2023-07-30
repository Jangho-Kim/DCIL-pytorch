import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return mnn.MaskConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return mnn.MaskConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


__all__ = ["wideresnet"]


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()



        # 0 -> part use, 1-> full use
        self.type_value = 0


        self.bn1_part = nn.BatchNorm2d(in_planes)
        self.bn1_full = nn.BatchNorm2d(in_planes)

        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes,stride )


        self.bn2_part = nn.BatchNorm2d(out_planes)
        self.bn2_full = nn.BatchNorm2d(out_planes)


        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)



        self.droprate = drop_rate
        self.equal_in_out = in_planes == out_planes



        self.conv_downsample = (
            (not self.equal_in_out)
            and conv1x1(
                in_planes,
                out_planes,stride
            )
            or None
        )

    def forward(self, x):
        if not self.equal_in_out:

            if self.type_value == 0 or self.type_value == 2:
             x = self.relu1(self.bn1_part(x))
            else:
             x = self.relu1(self.bn1_full(x))

        else:

            if self.type_value == 0 or self.type_value == 2:
                out = self.relu1(self.bn1_part(x))
            else:
                out = self.relu1(self.bn1_full(x))


        if self.type_value == 0 or self.type_value == 2:
            out = self.relu2(self.bn2_part(self.conv1(out if self.equal_in_out else x)))
        else:
            out = self.relu2(self.bn2_full(self.conv1(out if self.equal_in_out else x)))



        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equal_in_out else self.conv_downsample(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    drop_rate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, dataset, net_depth, widen_factor, drop_rate):
        super(WideResNet, self).__init__()

        # define fundamental parameters.
        self.dataset = dataset

        assert (net_depth - 4) % 6 == 0
        num_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        num_blocks = (net_depth - 4) // 6
        block = BasicBlock
        self.num_classes = self._decide_num_classes()
        self.type_value = 0


        # 1st conv before any network block
        self.conv1 = conv3x3( 3, num_channels[0] )



        # 1st block
        self.block1 = NetworkBlock(
            num_blocks, num_channels[0], num_channels[1], block, 1, drop_rate
        )
        # 2nd block
        self.block2 = NetworkBlock(
            num_blocks, num_channels[1], num_channels[2], block, 2, drop_rate
        )
        # 3rd block
        self.block3 = NetworkBlock(
            num_blocks, num_channels[2], num_channels[3], block, 2, drop_rate
        )

        # global average pooling and classifier


        self.bn1_part = nn.BatchNorm2d(num_channels[3])
        self.bn1_full= nn.BatchNorm2d(num_channels[3])


        self.relu = nn.ReLU(inplace=True)
        self.num_channels = num_channels[3]



        self.fc_part = nn.Linear(num_channels[3], self.num_classes)
        self.fc_full = nn.Linear(num_channels[3], self.num_classes)

        self._weight_initialization()

    def _decide_num_classes(self):
        if self.dataset == "cifar10" or self.dataset == "svhn":
            return 10
        elif self.dataset == "cifar100":
            return 100
        elif "imagenet" in self.dataset:
            return 1000

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x,type_value):

        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.type_value = type_value
            if isinstance(m, mnn.MaskConv2d):
                m.type_value = type_value

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        if type_value == 0 or type_value == 2:
            out = self.relu(self.bn1_part(out))
        else:
            out = self.relu(self.bn1_full(out))

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)

        # type 7 is sharing the fc
        if type_value == 0 or type_value == 2 or type_value == 7:
            out = self.fc_part(out)
        else:
            out = self.fc_full(out)


        return out


def wideresnet(data='cifar10', **kwargs):
    net_depth = int(kwargs.get('num_layers'))

    global mnn
    mnn = kwargs.get('mnn')
    assert mnn is not None, "Please specify proper pruning method"

    if "cifar" in data or "svhn" in data:
        model = WideResNet(
            dataset=data,
            net_depth=net_depth,
            widen_factor=int(kwargs.get('width_mult')),
            drop_rate=0.0,
        )
        image_size = 32
        return model, image_size
    else:
        raise NotImplementedError
