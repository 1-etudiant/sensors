import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
from .deformable_LKA import res2DF_LKA,DeformConv,DLKA_1


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]
        oup_inc = oup // 2
        # self.se1=SE_Block(oup_inc)
        # self.se2=SE_Block(inp)
        # self.se1=res2DF_LKA(oup_inc)
        #self.se2=res2DF_LKA(inp)
        # self.se1=deformable_LKA(oup_inc)
        # self.se2=deformable_LKA(inp)
        #self.dlka=DLKA_1(oup_inc)
        if self.benchmodel == 1:#不含下采样模块的
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                #pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.LeakyReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                #self.se1,
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.LeakyReLU(inplace=True),
                #self.dlka,
                #DeformConv(oup_inc,kernel_size=(5,5),stride=1,padding=2,groups=oup_inc,dilation=1),
            )
        else:#这是含有下采样模块的
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                #self.se2,
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.LeakyReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.LeakyReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                #self.se1,
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.LeakyReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)
    
class SE_Block(nn.Module):#se 模块
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = GlobalAvgPool2d()  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        #print(x.size())
        b, c, _, _= x.size()
        y = self.avg_pool(x).view(b, c) # squeeze 操作
        y = self.fc(y).view(b, c, 1, 1) # FC 获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上
class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=[2, 3], keepdim=True)
    
class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        #print(attn.size())
        attn = self.conv_spatial(attn)
        #print(attn.size())
        attn = self.conv1(attn)
        #print(attn.size())
        return u * attn
class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):#这里改模型宽度
        super(ShuffleNetV2, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [8, 4, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """groups is not supported for
                       1x1 Grouped Convolutions""")

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)#输入 3，输出 24
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):#[8 4 4]#先做 stage3
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))#1 个下采样模块
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))#多个无下采样模块
                input_channel = output_channel#第一层会改变通道数，后面的不会改变

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)#feature 存储 [stage3,stage4,stage2]

        self.index = self.stage_out_channels[2: 2 + len(self.stage_repeats)]
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        results = [None, None, None, None]
        for index, model in enumerate(self.features):
            x = model(x)
            # results.append(x)
            if index == 0:
                results[index] = x
            if x.size(1) in self.index:#通道数要对应
                position = self.index.index(x.size(1))  # Find the position in the index list
                results[position + 1] = x
        return results


def shufflenetv2(width_mult=1.):
    model = ShuffleNetV2(width_mult=width_mult)
    return model


if __name__ == "__main__":
    """Testing
    """
    model = ShuffleNetV2()
    print(model)