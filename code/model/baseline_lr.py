import torch.nn as nn
from model.rcan_video import RCAN_VIDEO
from model.flow_pwc import Flow_PWC
import torch


def make_model(args, parent=False):
    return BASELINE_LR(args)


class BASELINE_LR(nn.Module):
    def __init__(self, args):
        super(BASELINE_LR, self).__init__()

        n_colors = args.n_colors
        n_sequences = args.n_sequences
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        reduction = args.reduction
        res_scale = args.res_scale

        self.scale = args.scale

        self.pwcnet = Flow_PWC(load_pretrain=True, pretrain_fn=args.pwc_pretrain, device='cuda')
        self.sr_net = RCAN_VIDEO(n_colors, n_sequences, n_resgroups, n_resblocks, n_feats,
                                 reduction, self.scale, res_scale)

    def forward(self, x):
        if not x.ndimension() == 5:
            raise Exception("x.ndimension must equal 5: see x.ndimension={}".format(x.ndimension()))

        b, n, c, h, w = x.size()

        frame_list = [x[:, i, :, :, :] for i in range(n)]

        warp0_1, flow0_1 = self.pwcnet(frame_list[1], frame_list[0])
        warp2_1, flow2_1 = self.pwcnet(frame_list[1], frame_list[2])

        sr_output = self.sr_net(torch.stack((warp0_1, frame_list[1], warp2_1), dim=1))

        return sr_output

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


