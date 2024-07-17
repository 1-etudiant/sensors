import torch
import torchvision
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


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
    
#借鉴 res2net 改进：
class res2DF_LKA(nn.Module):
    def __init__(self,dim,scale=4,stride=1,groups=1):
        super(res2DF_LKA,self).__init__()
        self.scale=scale
        self.width=dim//scale
        #self.conv1=Conv(dim,dim,k=1,s=1,act=False)
        self.conv2=DeformConv(self.width,kernel_size=(3,3),padding=1,groups=self.width)
        self.conv3=DeformConv(self.width*2, kernel_size=(5,5), padding=2, groups=self.width*2)
        self.conv4 = DeformConv(self.width*3, kernel_size=(7,7), stride=1, padding=9, groups=self.width*3, dilation=3)
        # self.conv2=DeformConv(self.width,kernel_size=(5,5),padding=1,groups=self.width)
        # self.conv3=DeformConv(self.width*2, kernel_size=(7, 7), padding=2, groups=self.width*2)
        # self.conv4 = DeformConv(self.width*3, kernel_size=(9, 9), stride=1, padding=9, groups=self.width*3, dilation=3)
        #self.convs=nn.ModuleList([Conv(self.width,self.width,k=kernel_size,s=stride,p=kernel_size//2,g=groups) for kernel_size in [3,5,7]])
        self.conv5=nn.Conv2d(dim,dim,kernel_size=1,stride=1)
    
    def forward(self,x):
        u=x.clone()
        #x1=self.conv1(x)
        xs=torch.chunk(x,self.scale,1)#split
        y=xs[0]#2
        #print(xs[1].shape)
        y1=self.conv2(xs[1])#2
        y2=self.conv3(torch.cat((y1,xs[2]),1))#4
        y3=self.conv4(torch.cat((y2,xs[3]),1))#6
        x=self.conv5(torch.cat((xs[0],y3),1))#8
        return u*x
class split_DLKA(nn.Module):
    def __init__(self, cin,scale=2):
        super(split_DLKA,self).__init__()
        self.scale=scale
        self.width=cin//scale
        self.conv1=DeformConv(self.width,kernel_size=(5,5),padding=2,groups=self.width)
        self.conv2=DeformConv(self.width,kernel_size=(7,7),padding=9,groups=self.width,dilation=3)
        self.conv3=nn.Conv2d(cin,cin,kernel_size=1,stride=1)
    def forward(self,x):
        u=x.clone()
        xs=torch.chunk(x,self.scale,1)
        y1=self.conv1(xs[0])
        y2=self.conv2(xs[1])
        y=self.conv3(torch.cat((y1,y2),1))
        return y*u




class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        #self.Attention = deformable_LKA(c_)
        self.Attention = res2DF_LKA(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        #return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
        return x +self.cv2(self.Attention(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
class C3_DLKA(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    

class LKA(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        #self.Attention = deformable_LKA(c_)
        self.Attention=res2DF_LKA(c_)#自己改进的
        self.add = shortcut and c1 == c2

    def forward(self, x):
        #return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
        return x +self.cv2(self.Attention(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
    

def main():
    x=torch.randn(8,8,256,256)
    filter=res2DF_LKA(8)
    x=filter(x)
    print(x.shape)
if __name__ =="__main__":
    main()