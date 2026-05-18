import torch
import numpy as np
from PIL import Image

import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.autograd import Variable
from torchvision import transforms
from skimage.color import rgb2hsv
from utils.image_io import np_to_torch, torch_to_np
import pdb



class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3, gpu_ids=None):  # only work for gpu case at this moment
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        self.boxfilter = nn.AvgPool2d(kernel_size=2 * self.r + 1, stride=1, padding=self.r)

    def forward(self, I, p):
        """
        I -- guidance image, should be [0, 1]
        p -- filtering input image, should be [0, 1]
        """

        # N = self.boxfilter(self.tensor(p.size()).fill_(1))
        N = self.boxfilter(torch.ones(p.size()))

        if I.is_cuda:
            N = N.cuda()

        # print(N.shape)
        # print(I.shape)
        # print('-----------')

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I * p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I * I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b

def DCLoss(img, patch_size):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
    dc = maxpool(0-img[:, :, :, :])

    # r = 50
    # eps = 1e-3
    # guided_filter = GuidedFilter(r=r, eps=eps)
    # if img.shape[1] > 1:
    #     # rgb2gray
    #     guidance = 0.2989 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.1140 * img[:, 2, :, :]
    # else:
    #     guidance = img
    # # rescale to [0,1]
    # guidance = (guidance + 1) / 2
    # guidance = torch.unsqueeze(guidance, dim=1)
    # dc = guided_filter(guidance, dc)
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()) 
     
    loss = L1Loss(size_average=True)(-dc, target)
    return loss

mse = MSELoss()
def SV_loss(img):
    hsv = np_to_torch(rgb2hsv(torch_to_np(img).transpose(1, 2, 0)))
    cap_prior = hsv[:, :, :, 2] - hsv[:, :, :, 1]

    cap_loss = mse(cap_prior, torch.zeros_like(cap_prior))
    return cap_loss

def BCLoss(img, patch_size):
    """
    calculating bright channel of image, the image shape is of N*C*W*H
    """
    patch_size = 35
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))

    dc = maxpool(img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()+1) 
    loss = L1Loss(size_average=False)(dc, target)
    return loss
    
if __name__=="__main__":
    img = Image.open('clear.jpg')
    totensor = transforms.ToTensor()
    
    img = totensor(img)
    
    img = Variable(img[None, :, :, :].cuda(), requires_grad=True)    
    loss = DCLoss(img, 35)
    
    # loss.backward()



    



