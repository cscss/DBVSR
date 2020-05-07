import math
import torch.nn as nn


def make_model(args):
    return RCAN_VIDEO(n_colors=args.n_colors, n_sequence=args.n_sequence, n_resgroups=args.n_resgroups,
                      n_resblocks=args.n_resblocks, n_feat=args.n_feat, reduction=args.reduction,
                      scale=args.scale, res_scale=args.res_scale)


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, res_scale, n_resblocks, act=nn.ReLU(True)):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Upsampler
class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feat, bn=False, act=False):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feat, 4 * n_feat, kernel_size=3, stride=1, padding=1))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(nn.ReLU(True))
        elif scale == 3:
            m.append(nn.Conv2d(n_feat, 9 * n_feat, kernel_size=3, stride=1, padding=1))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(nn.ReLU(True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class RCAN_VIDEO(nn.Module):
    def __init__(self, n_colors, n_sequence, n_resgroups, n_resblocks, n_feat, reduction=16, scale=1, res_scale=1):
        super(RCAN_VIDEO, self).__init__()
        print("Creating RCAN VIDEO Net")

        # define head module
        modules_head = [nn.Conv2d(n_colors * n_sequence, n_feat, kernel_size=3, stride=1, padding=1)]

        # define body module
        modules_body = [
            ResidualGroup(
                n_feat, kernel_size=3, reduction=reduction, act=nn.ReLU(True), res_scale=res_scale,
                n_resblocks=n_resblocks)
            for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1))

        # define tail module
        modules_tail = [Upsampler(scale, n_feat),
                        nn.Conv2d(n_feat, n_colors, kernel_size=3, stride=1, padding=1)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        b, n, c, h, w = x.size()
        x = x.view(b, n * c, h, w)

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
