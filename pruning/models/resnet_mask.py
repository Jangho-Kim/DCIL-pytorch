"""ResNet/WideResNet in PyTorch.
See the paper "Deep Residual Learning for Image Recognition"
(https://arxiv.org/abs/1512.03385)
"""
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return mnn.MaskConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return mnn.MaskConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample_conv=None,downsample_p=None,downsample_f=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")


        # 0 -> part use, 1-> full use
        self.type_value = 0

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1_part = norm_layer(planes)
        self.bn1_full = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)


        self.conv2 = conv3x3(planes, planes)
        self.bn2_part = norm_layer(planes)
        self.bn2_full = norm_layer(planes)

        self.downsample_conv = downsample_conv
        self.downsample_p = downsample_p
        self.downsample_f = downsample_f

        self.stride = stride



    def forward(self, x):
        identity = x

        out = self.conv1(x)

        # switch the bn
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn1_part(out)
        else:
            out = self.bn1_full(out)


        out = self.relu(out)
        out = self.conv2(out)

        # switch the bn
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn2_part(out)
        else:
            out = self.bn2_full(out)


        if self.downsample_conv is not None:

            if self.type_value == 0 or self.type_value == 2:
                temp = self.downsample_conv(x)
                identity = self.downsample_p(temp)
            else:
                temp = self.downsample_conv(x)
                identity = self.downsample_f(temp)


        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample_conv=None,downsample_p=None,downsample_f=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d



        # 0 -> part use, 1-> full use
        self.type_value = 0

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv1x1(inplanes, width)
        self.bn1_part = norm_layer(width)
        self.bn1_full = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2_part = norm_layer(width)
        self.bn2_full = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3_part = norm_layer(planes * self.expansion)
        self.bn3_full = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample_conv = downsample_conv
        self.downsample_p = downsample_p
        self.downsample_f = downsample_f


        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # switch the bn
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn1_part(out)
        else:
            out = self.bn1_full(out)
        out = self.relu(out)


        out = self.conv2(out)
        # switch the bn
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn2_part(out)
        else:
            out = self.bn2_full(out)
        out = self.relu(out)


        out = self.conv3(out)
        # switch the bn
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn3_part(out)
        else:
            out = self.bn3_full(out)


        if self.downsample_conv is not None:

            if self.type_value == 0 or self.type_value == 2:
                temp = self.downsample_conv(x)
                identity = self.downsample_p(temp)
            else:
                temp = self.downsample_conv(x)
                identity = self.downsample_f(temp)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.block_name = str(block.__name__)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3: 
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = mnn.MaskConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                    bias=False)



        self.bn1_part = norm_layer(self.inplanes)
        self.bn1_full = norm_layer(self.inplanes)


        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



        self.fc_part = nn.Linear(512 * block.expansion, num_classes)
        self.fc_full = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, mnn.MaskConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer


        downsample_conv = None
        downsample_p = None
        downsample_f = None

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_conv = conv1x1(self.inplanes, planes * block.expansion, stride)
            downsample_p = norm_layer(planes * block.expansion)
            downsample_f = norm_layer(planes * block.expansion)




        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample_conv,downsample_p,downsample_f, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x,type_value):
        # See note [TorchScript super()]

        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.type_value = type_value
            if isinstance(m, Bottleneck):
                m.type_value = type_value
            if isinstance(m, mnn.MaskConv2d):
                m.type_value = type_value

        x = self.conv1(x)

        if type_value == 0 or type_value == 2:
            x = self.bn1_part(x)
        else:
            x = self.bn1_full(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # type 7 is sharing the fc
        if type_value == 0 or type_value == 2 or type_value == 7:
            x = self.fc_part(x)
        else:
            x = self.fc_full(x)

        return x

    def forward(self, x,type_value):
        return self._forward_impl(x,type_value)

class ResNet_CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_CIFAR, self).__init__()
        self.block_name = str(block.__name__)



        if norm_layer is None:
            norm_layer = nn.BatchNorm2d


        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3: 
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = mnn.MaskConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                    bias=False)                               

        self.bn1_part = norm_layer(self.inplanes)
        self.bn1_full = norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_part = nn.Linear(64 * block.expansion, num_classes)
        self.fc_full = nn.Linear(64 * block.expansion, num_classes)



        for m in self.modules():
            if isinstance(m, mnn.MaskConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer


        downsample_conv = None
        downsample_p = None
        downsample_f = None

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_conv = conv1x1(self.inplanes, planes * block.expansion, stride)
            downsample_p = norm_layer(planes * block.expansion)
            downsample_f = norm_layer(planes * block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample_conv,downsample_p,downsample_f, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)




    # type value 0 -> pruned and update only important
    # type value 0 -> pruned and update only important
    # type value 0 -> pruned and update only important


    def _forward_impl(self, x, type_value):
        # See note [TorchScript super()]

        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.type_value = type_value
            if isinstance(m, mnn.MaskConv2d):
                m.type_value = type_value
            if isinstance(m, Bottleneck):
                m.type_value = type_value



        x = self.conv1(x)


        if type_value == 0 or type_value == 2:
            x = self.bn1_part(x)
        else:
            x = self.bn1_full(x)


        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)


        # type 7 is sharing the fc
        if type_value == 0 or type_value == 2 or type_value == 7:
            x = self.fc_part(x)
        else:
            x = self.fc_full(x)

        return x

    def forward(self, x,type_value):
        return self._forward_impl(x,type_value)


# Model configurations
cfgs = {
    '18':  (BasicBlock, [2, 2, 2, 2]),
    '34':  (BasicBlock, [3, 4, 6, 3]),
    '50':  (Bottleneck, [3, 4, 6, 3]),
    '101': (Bottleneck, [3, 4, 23, 3]),
    '152': (Bottleneck, [3, 8, 36, 3]),
}
cfgs_cifar = {
    '20':  [3, 3, 3],
    '32':  [5, 5, 5],
    '44':  [7, 7, 7],
    '56':  [9, 9, 9],
    '110': [18, 18, 18],
}


def resnet(data='cifar10', **kwargs):
    r"""ResNet models from "[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)"
    Args:
        data (str): the name of datasets
    """
    num_layers = str(kwargs.get('num_layers'))

    # set pruner
    global mnn
    mnn = kwargs.get('mnn')
    assert mnn is not None, "Please specify proper pruning method"

    if data in ['cifar10', 'cifar100']:
        if num_layers in cfgs_cifar.keys():
            if int(num_layers) >= 44:
                model = ResNet_CIFAR(Bottleneck, cfgs_cifar[num_layers], int(data[5:]))
            else:
                model = ResNet_CIFAR(BasicBlock, cfgs_cifar[num_layers], int(data[5:]))
        else:
            model = None
        image_size = 32
    elif data == 'imagenet':
        if num_layers in cfgs.keys():
            block, layers = cfgs[num_layers]
            model = ResNet(block, layers, 1000)
        else:
            model = None
        image_size = 224
    else:
        model = None
        image_size = None

    return model, image_size