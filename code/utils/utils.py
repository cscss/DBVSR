import random
import torch
import torch.nn.functional as F
import numpy as np
import math
from skimage import color as sc


def get_patch(*args, patch_size=17, scale=1):
    """
    Get patch from an image
    """
    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret


def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = sc.rgb2ycbcr(img)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=1, n_colors=3):
    def _np2Tensor(img):
        if img.shape[2] == 3 and n_colors == 3:
            # mean_RGB = np.array([123.68, 116.779, 103.939])
            img = img.astype('float64')
        elif img.shape[2] == 3 and n_colors == 1:
            mean_YCbCr = np.array([109, 0, 0])
            img = img.astype('float64') - mean_YCbCr

        # NHWC -> NCHW
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]


def data_augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = np.rot90(img)

        return img

    return [_augment(a) for a in args]


def postprocess(*images, rgb_range, ycbcr_flag, device):
    def _postprocess(img, rgb_coefficient, ycbcr_flag, device):
        if ycbcr_flag:
            mean_YCbCr = torch.Tensor([109]).to(device)
            out = (img.mul(rgb_coefficient) + mean_YCbCr).clamp(16, 235)
        else:
            # mean_RGB = torch.Tensor([123.68, 116.779, 103.939]).to(device)
            # mean_RGB = mean_RGB.reshape([1, 3, 1, 1])
            out = (img.mul(rgb_coefficient)).clamp(0, 255).round()

        return out

    rgb_coefficient = 255 / rgb_range
    return [_postprocess(img, rgb_coefficient, ycbcr_flag, device) for img in images]


def psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calc_psnr(gt, sr, rgb_range=255, shave=4, is_rgb=False):
    gt_in = gt[:, :, shave:-shave, shave:-shave]
    sr_in = sr[:, :, shave:-shave, shave:-shave]
    gt_in = gt_in[0, :, :, :]
    sr_in = sr_in[0, :, :, :]

    gt_np = np.transpose(gt_in.detach().cpu().numpy(), (1, 2, 0))
    sr_np = np.transpose(sr_in.detach().cpu().numpy(), (1, 2, 0))

    PSNR = psnr(gt_np * (255 / rgb_range), sr_np * (255 / rgb_range))

    return PSNR


def calc_grad_sobel(img, device):
    if not isinstance(img, torch.Tensor):
        raise Exception("Now just support torch.Tensor. See the Type(img)={}".format(type(img)))
    if not img.ndimension() == 4:
        raise Exception("Tensor ndimension must equal to 4. See the img.ndimension={}".format(img.ndimension()))

    img = torch.mean(img, dim=1, keepdim=True)

    sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_X = torch.from_numpy(sobel_filter_X).float().to(device)
    sobel_filter_Y = torch.from_numpy(sobel_filter_Y).float().to(device)
    grad_X = F.conv2d(img, sobel_filter_X, bias=None, stride=1, padding=1)
    grad_Y = F.conv2d(img, sobel_filter_Y, bias=None, stride=1, padding=1)
    grad = torch.sqrt(grad_X.pow(2) + grad_Y.pow(2))

    return grad_X, grad_Y, grad
