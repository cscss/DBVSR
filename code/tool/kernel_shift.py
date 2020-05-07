'''
function psf = adjust_psf_center(psf)

[X Y] = meshgrid(1:size(psf,2), 1:size(psf,1));
xc1 = sum2(psf .* X);
yc1 = sum2(psf .* Y);
xc2 = (size(psf,2)+1) / 2;
yc2 = (size(psf,1)+1) / 2;
xshift = round(xc2 - xc1);
yshift = round(yc2 - yc1);
psf = warpimage(psf, [1 0 -xshift; 0 1 -yshift]);

function val = sum2(arr)
val = sum(arr(:));
%%
% M should be an inverse transform!
function warped = warpimage(img, M)
if size(img,3) == 3
    warped(:,:,1) = warpProjective2(img(:,:,1), M);
    warped(:,:,2) = warpProjective2(img(:,:,2), M);
    warped(:,:,3) = warpProjective2(img(:,:,3), M);
    warped(isnan(warped))=0;
else
    warped = warpProjective2(img, M);
    warped(isnan(warped))=0;
end

%%
function result = warpProjective2(im,A)
%
% function result = warpProjective2(im,A)
%
% im: input image
% A: 2x3 affine transform matrix or a 3x3 matrix with [0 0 1]
% for the last row.
% if a transformed point is outside of the volume, NaN is used
%
% result: output image, same size as im
%

if (size(A,1)>2)
  A=A(1:2,:);
end

% Compute coordinates corresponding to input
% and transformed coordinates for result
[x,y]=meshgrid(1:size(im,2),1:size(im,1));
coords=[x(:)'; y(:)'];
homogeneousCoords=[coords; ones(1,prod(size(im)))];
warpedCoords=A*homogeneousCoords;
xprime=warpedCoords(1,:);%./warpedCoords(3,:);
yprime=warpedCoords(2,:);%./warpedCoords(3,:);

result = interp2(x,y,im,xprime,yprime, 'linear');
result = reshape(result,size(im));

return;


'''
import numpy as np
import cv2
from tool.interp2 import interp2linear
import torch


def warpimage(kernel, A):
    kernel_size = kernel.shape[0]

    arrange = np.arange(1, kernel_size + 1)
    x, y = np.meshgrid(arrange, arrange)
    coords = np.zeros((2, kernel_size * kernel_size), dtype=float)
    x_col = x.reshape((kernel_size * kernel_size, 1))
    y_col = y.reshape((kernel_size * kernel_size, 1))
    coords[0, :] = y_col.T
    coords[1, :] = x_col.T  # coords  2, 225
    homogeneousCoords = np.ones((3, kernel_size * kernel_size))
    homogeneousCoords[0:2, :] = coords
    warpedCoords = np.matmul(A, homogeneousCoords)

    xprime = warpedCoords[0, :]
    yprime = warpedCoords[1, :]

    result = interp2linear(kernel, xprime - 1, yprime - 1)

    # result = interp2(x, y, im, xprime, yprime, 'linear')
    result = np.reshape(result, kernel.shape);
    return result

    # one = np.ones((1,))


def kernel_shift(kernel):
    kernel_size = kernel.shape[0]
    arrange = np.arange(1, kernel_size + 1)
    X, Y = np.meshgrid(arrange, arrange)
    xc1 = np.sum(np.multiply(kernel, X))
    yc1 = np.sum(np.multiply(kernel, Y))
    xc2 = (kernel_size + 1) / 2
    yc2 = (kernel_size + 1) / 2
    xshift = round(xc2 - xc1)
    yshift = round(yc2 - yc1)
    A = np.array([[1, 0, -xshift], [0, 1, -yshift]], dtype=float)
    kernel_after = warpimage(kernel, A)
    return kernel_after


if __name__ == '__main__':
    # k = torch.load('./kernelX4.pt')
    # k = np.array(k, dtype=float)
    # print(np.sum(k))
    # kernel_after = kernel_shift(k)
    # kernel_after = kernel_after/np.sum(kernel_after)
    # kernel_after_save = torch.FloatTensor(kernel_after).cuda()
    # torch.save(kernel_after_save, './kernelX4_after.pt')
    k = cv2.imread('/home/cheng/study/experiment_code/code_Kernel/experiment/kernel_15X3/result/0801_kernel1.png')
    k = k[:, :, 0]
    k = k / np.sum(k)
    kernel_after = kernel_shift(k)
    # print(k)
    # print(kernel_after)
    print(k[9, 9] - kernel_after[8, 8])
    # print(np.sum(kernel_after))
    # cv2.imwrite('./afterX3.png', kernel_after*255)
    # a = cv2.imread('./after.png')
    # print(k.shape)
    print(kernel_after)
    # print('asdf')
    # print(np.sum(k))
    # print(np.sum(kernel_after))
