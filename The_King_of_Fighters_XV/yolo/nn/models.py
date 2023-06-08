import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_anchors, dist2bbox



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


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

# heads
class Detect(nn.Module):
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=2, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = 3  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        # c2 是bboxloss，c3是clsloss
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format == 'edgetpu':  # FlexSplitV ops issue
            x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m_conf, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for mi, s in zip(m.m_cls, m.stride):  # from
            b = mi[-1].bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

class YOLO(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone_1 = nn.Sequential(
            Conv(3, 16, 3, 2),
            Conv(16,32, 3, 2),
            C2f(32,32,1,True),

            Conv(32,64,3,2),
            C2f(64,64,2,True)
        )
        self.backbone_2 = nn.Sequential(
            Conv(64,128,3,2),
            C2f(128,128,2,True)
        )
        self.backbone_3 = nn.Sequential(
            Conv(128,256,3,2),
            C2f(256,256,1,True)
        )
        self.sppf = SPPF(256,256,5)

        #     YOLOv8n backbone
        #     [from, repeats, module, args]
        #   - [-1, 1, Conv, [16, 3, 2]]  # 0-P1/2
        #   - [-1, 1, Conv, [32, 3, 2]]  # 1-P2/4
        #   - [-1, 1, C2f, [32, True]]
        #   - [-1, 1, Conv, [64, 3, 2]]  # 3-P3/8
        #   - [-1, 2, C2f, [64, True]]
        #   - [-1, 1, Conv, [128, 3, 2]]  # 5-P4/16
        #   - [-1, 2, C2f, [128, True]]
        #   - [-1, 1, Conv, [256, 3, 2]]  # 7-P5/32
        #   - [-1, 1, C2f, [256, True]]
        #   - [-1, 1, SPPF, [256, 5]]  # 9

        #     YOLOv8n  head:
        #   - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
        #   - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
        #   - [-1, 1, C2f, [128]]  # 12

        #   - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
        #   - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
        #   - [-1, 1, C2f, [64]]  # 15 (P3/8-small)

        #   - [-1, 1, Conv, [256, 3, 2]]
        #   - [[-1, 12], 1, Concat, [1]]  # cat head P4
        #   - [-1, 1, C2f, [128]]  # 18 (P4/16-medium)

        #   - [-1, 1, Conv, [128, 3, 2]]
        #   - [[-1, 9], 1, Concat, [1]]  # cat head P5
        #   - [-1, 1, C2f, [256]]  # 21 (P5/32-large)

        #   - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
        self.unsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.concat = Concat()
        
        # self.head_c2f*
        self.head_c2f_0 = C2f(384, 128, 1)
        self.head_c2f_1 = C2f(192, 64, 1)
        self.head_c2f_2 = C2f(384, 256, 1)

        # self.head_conv*
        self.head_conv0 = Conv(64, 256, 3, 2)
        self.head_conv1 = Conv(128, 128, 3, 2)

        # self.detect_head
        self.detect = Detect(nc=2, ch=(64,128,256))

    def forward(self, x):
        x1 = self.backbone_1(x)
        x2 = self.backbone_2(x1)
        x3 = self.backbone_3(x2)

        x4 = self.sppf(x3)

        print('shlskj')
        x5 = self.unsample(x4)
        x5 = self.concat([x5,x2])
        x5 = self.head_c2f_0(x5)

        x6 = self.unsample(x5)
        x6 = self.concat([x6,x1])
        x6 = self.head_c2f_1(x6)

        x7 = self.head_conv0(x6)
        x7 = self.concat([x5, x7])
        x7 = self.head_c2f_0(x7)

        x8 = self.head_conv1(x7)
        x8 = self.concat([x8, x4])
        x8 = self.head_c2f_2(x8)

        
        output = self.detect([x6,x7,x8])

        return output
    
if __name__ == "__main__":
    model_path = 'D:\\Work_space\\Projects\\AI-Game\\The_King_of_Fighters_XV\\model\\best.pt'

    # 加载预训练权重
    checkpoint = torch.load(model_path)

    # Step 1: Define the model architecture
    model = YOLO()

    # Step 2: Load the pre-trained weights
    model.load_state_dict(checkpoint['model'])

    print(model)
    