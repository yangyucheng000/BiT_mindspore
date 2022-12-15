import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from collections import OrderedDict  # pylint: disable=g-importing-member
from mindspore.common.initializer import initializer, Zero

# just for test
#ms.set_context(device_target="Ascend")
ms.context.set_context(device_target="Ascend")
# ms.context.set_context(device_id=1)

class StdConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros', data_format='NCHW'):
        super().__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation, group, has_bias, weight_init, bias_init, data_format)
        self.ops_conv2d = ops.Conv2D(self.out_channels, self.kernel_size,
                                 pad_mode=self.pad_mode, pad=self.padding, stride=self.stride,)
        self.reduce_mean = ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        w = self.weight
        m = self.reduce_mean(w, (1, 2, 3))
        w = w - m
        v = self.reduce_mean(ops.square(w), (1, 2, 3))
        w = w / ops.sqrt(v + 1e-10)
        # bias does not matter when combined with GN
        #"dilation","data_format" maybe can be removed
        ops_conv2d = ops.Conv2D(self.out_channels, self.kernel_size, pad_mode=self.pad_mode,
                                pad=self.padding, stride=self.stride, dilation=self.dilation,
                                group=self.group, data_format=self.format)
        return  ops_conv2d(x, w)
    # def construct(self, x):
    #     w = self.weight
    #     v = w.var((1, 2, 3), keepdims=True)
    #     m = w.mean((1, 2, 3), keep_dims=True)
    #     w = (w - m) / ops.sqrt(v + 1e-10)

    #     return self.conv2d(x, w)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, group=groups, has_bias=bias)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, has_bias=bias)

class AdaptiveAvgPool2d_layer(nn.Cell):
    
    def construct(self, x):
        # ops_AdaptiveAvgPool2D = ops.AdaptiveAvgPool2D(self.output_size)
        mean = ops.ReduceMean(keep_dims=True)
        out = mean(x, (2, 3))
        return out

def tf2th(conv_weights):
  """Possibly convert HWIO to OIHW."""
  if conv_weights.ndim == 4:
    conv_weights = conv_weights.transpose([3, 2, 0, 1])
  return conv_weights.asnumpy()

class PreActBottleneck(nn.Cell):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU()
        self.downsample = None

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride)
    
    def construct(self, x):
        out = self.relu(self.gn1(x))

        # residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(out)
        
        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual

class ResNetV2(nn.Cell):
    """Implementation of Pre-activation (v2) ResNet mode."""
    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        self.root = nn.SequentialCell(OrderedDict([
            ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, pad_mode="pad",has_bias=False)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2)),
            # The following is subtly not the same!
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.body = nn.SequentialCell(OrderedDict([
            ('block1', nn.SequentialCell(OrderedDict(
                [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf)) for i in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.SequentialCell(OrderedDict(
                [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.SequentialCell(OrderedDict(
                [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
            ))),
            ('block4', nn.SequentialCell(OrderedDict(
                [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
            ))),
        ]))

        self.zero_head = zero_head
        self.head = nn.SequentialCell(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048*wf)),
            ('relu', nn.ReLU()),
            ('avg', AdaptiveAvgPool2d_layer()),
            ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, has_bias=True, weight_init="zeros")),
        ]))

    
    def construct(self, x):
        x = self.head(self.body(self.root(x)))
        # assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[...,0,0]


KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])


# for test
data1 = ms.Tensor(np.ones([1, 4, 8, 8]), ms.float32)
data2 = ms.Tensor(np.zeros([1, 256, 56, 56]), ms.float32)
test_data = ms.Tensor(np.zeros([1, 3, 112, 112]), ms.float32)
resnet = KNOWN_MODELS['BiT-M-R50x1']()

