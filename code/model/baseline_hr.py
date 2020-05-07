from model.flow_pwc import Flow_PWC
import torch
from model.modules import *


def make_model(args, parent=False):
    return BASELINE_HR(args)

class BASELINE_HR(nn.Module):
    def __init__(self, args):
        super(BASELINE_HR, self).__init__()

        n_colors = args.n_colors
        n_sequences = args.n_sequences
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        reduction = args.reduction
        res_scale = args.res_scale

        self.scale = args.scale

        self.pwcnet = Flow_PWC(load_pretrain=True, pretrain_fn=args.pwc_pretrain, device='cuda')

        # define head module
        modules_head = [
            nn.Conv2d(n_sequences * (n_colors * (self.scale ** 2 + 1)), n_feats, kernel_size=3, stride=1, padding=1)]

        # define body module
        modules_body = [
            ResidualGroup(
                n_feats, kernel_size=3, reduction=reduction, act=nn.ReLU(False), res_scale=res_scale,
                n_resblocks=n_resblocks)
            for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))

        # define tail module
        modules_tail = [Upsampler(self.scale, n_feats),
                        nn.Conv2d(n_feats, n_colors, kernel_size=3, stride=1, padding=1)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def spatial2depth(self, spatial, scale):
        depth_list = []
        for i in range(scale):
            for j in range(scale):
                depth_list.append(spatial[:, :, i::scale, j::scale])
        depth = torch.cat(depth_list, dim=1)
        return depth

    def forward(self, x, x_bicubic, kernel):
        kernel = kernel[0]
        if not x.ndimension() == 5:
            raise Exception("x.ndimension must equal 5: see x.ndimension={}".format(x.ndimension()))

        b, n, c, h, w = x.size()
        kernel_size, _ = kernel.size()
        frame_list = [x[:, i, :, :, :] for i in range(n)]
        bicubic_list = [x_bicubic[:, i, :, :, :] for i in range(n)]
        x_mid_bicubic = bicubic_list[n//2]

        # pwc for flow and warp
        warp0_1, flow0_1 = self.pwcnet(frame_list[1], frame_list[0])
        warp2_1, flow2_1 = self.pwcnet(frame_list[1], frame_list[2])

        bic_warp0_1, bic_flow0_1 = self.pwcnet(bicubic_list[1], bicubic_list[0])
        bic_warp2_1, bic_flow2_1 = self.pwcnet(bicubic_list[1], bicubic_list[2])

        bic_warp0_1_depth = self.spatial2depth(bic_warp0_1, scale=self.scale)
        bic_warp2_1_depth = self.spatial2depth(bic_warp2_1, scale=self.scale)
        x_mid_bicubic_depth = self.spatial2depth(x_mid_bicubic, scale=self.scale)

        sr_input = torch.cat((bic_warp0_1_depth, warp0_1, x_mid_bicubic_depth, frame_list[1], bic_warp2_1_depth, warp2_1), dim=1)

        # RCAN super-resolution with upsample
        head_out = self.head(sr_input)
        body_out = self.body(head_out)
        sr_output = self.tail(body_out + head_out)

        return  sr_output

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
