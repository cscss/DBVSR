import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def make_model(args):
    return Deconv_Layer()


class Deconv_Layer(nn.Module):

    def __init__(self, scale=4, device='cuda', we=0.02):
        super(Deconv_Layer, self).__init__()
        print("Creating Deconv layer")
        self.device = device
        self.scale = scale
        self.we = we

        fx = np.array([[0, -1, 1],
                       [0, 0, 0],
                       [0, 0, 0]])
        fy = np.array([[0, 0, 0],
                       [-1, 0, 0],
                       [1, 0, 0]])
        self.fx = torch.from_numpy(fx).view(1, 1, 3, 3).to(self.device)
        self.fy = torch.from_numpy(fy).view(1, 1, 3, 3).to(self.device)

    def warp_by_flow(self, x, flo, device='cuda'):
        B, C, H, W = flo.size()

        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(device)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, padding_mode='border')

        return output

    def convert_psf2otf(self, ker, size=(1, 1, 240, 200)):
        psf = torch.zeros(size).cuda()
        # circularly shift
        centre = ker.shape[2] // 2 + 1
        psf[:, :, :centre, :centre] = ker[:, :, (centre - 1):, (centre - 1):]
        psf[:, :, :centre, -(centre - 1):] = ker[:, :, (centre - 1):, :(centre - 1)]
        psf[:, :, -(centre - 1):, :centre] = ker[:, :, :(centre - 1), (centre - 1):]
        psf[:, :, -(centre - 1):, -(centre - 1):] = ker[:, :, :(centre - 1), :(centre - 1)]
        # compute the otf
        otf = torch.rfft(psf, 3, onesided=False)
        return otf

    def inv_fft_kernel_est(self, ker_f, fx_f, fy_f, we):
        inv_fxy_f = fx_f[:, :, :, :, 0] * fx_f[:, :, :, :, 0] + fx_f[:, :, :, :, 1] * fx_f[:, :, :, :, 1] + \
                    fy_f[:, :, :, :, 0] * fy_f[:, :, :, :, 0] + fy_f[:, :, :, :, 1] * fy_f[:, :, :, :, 1]

        inv_denominator = ker_f[:, :, :, :, 0] * ker_f[:, :, :, :, 0] + ker_f[:, :, :, :, 1] * ker_f[:, :, :, :, 1] + \
                          we * inv_fxy_f

        # pseudo inverse kernel in flourier domain.
        inv_ker_f = torch.zeros_like(ker_f)
        inv_ker_f[:, :, :, :, 0] = ker_f[:, :, :, :, 0] / inv_denominator
        inv_ker_f[:, :, :, :, 1] = -ker_f[:, :, :, :, 1] / inv_denominator
        return inv_ker_f

    def deconv(self, inv_ker_f, _input_blur):
        fft_input_blur = torch.rfft(_input_blur, 3, onesided=False).cuda()
        # delement-wise multiplication.
        deblur_f = torch.zeros_like(inv_ker_f).cuda()
        deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] \
                                  - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
        deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
                                  + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]

        deblur = torch.irfft(deblur_f, 3, onesided=False)
        return deblur_f, deblur

    def spatial2depth(self, spatial, scale):
        depth_list = []
        for i in range(scale):
            for j in range(scale):
                depth_list.append(spatial[:, :, i::scale, j::scale])
        depth = torch.cat(depth_list, dim=1)
        return depth

    def forward(self, mid_frame_bic, kernel):
        b, c, up_h, up_w = mid_frame_bic.size()

        # deconv
        _, _, k, _ = kernel.size()
        pad_size = 2 * k
        mid_frame_bic_pad = F.pad(mid_frame_bic, pad=(pad_size, pad_size, pad_size, pad_size), mode='replicate')

        kernel_f = self.convert_psf2otf(kernel, (b, 1, up_h + pad_size * 2, up_w + pad_size * 2))
        fx_f = self.convert_psf2otf(self.fx, (b, 1, up_h + pad_size * 2, up_w + pad_size * 2))
        fy_f = self.convert_psf2otf(self.fy, (b, 1, up_h + pad_size * 2, up_w + pad_size * 2))

        re = []
        in_k_f = self.inv_fft_kernel_est(kernel_f, fx_f, fy_f, self.we)
        for i in range(c):
            deconv_f, deconv_r = self.deconv(in_k_f, mid_frame_bic_pad[:, i:i + 1, :, :])
            re.append(deconv_r)
        HR_deconv = torch.cat(re, dim=1)
        HR_deconv = HR_deconv[:, :, pad_size:-pad_size, pad_size:-pad_size]

        return HR_deconv
