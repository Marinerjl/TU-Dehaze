import torch.nn as nn
from models.DCPDehazeGenerator import atmospheric_light
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Trans_fineNet(nn.Module):
    def __init__(self):
      super(Trans_fineNet, self).__init__()
      Trans_coarseNet_conv_1 = nn.Conv2d(in_channels=3,
                             out_channels=5,
                             kernel_size=11,
                             stride=1,
                             padding=11>>1,
                             bias=True)
      Trans_coarseNet_conv_2 = nn.Conv2d(5, 5, 9, 1, 9 >> 1, bias=True)
      Trans_coarseNet_conv_3 = nn.Conv2d(5, 10, 7, 1, 7 >> 1, bias=True)
      Trans_coarseNet_conv_4 = nn.Conv2d(10, 1, 1, 1, padding=1 >> 1, bias=True)
      Trans_coarseNet_Conv = [Trans_coarseNet_conv_1, Trans_coarseNet_conv_2, Trans_coarseNet_conv_3, Trans_coarseNet_conv_4]

      self.condition_conv = nn.Sequential(*Trans_coarseNet_Conv)
      self.conv1 = nn.Conv2d(in_channels=3,
                             out_channels=4,
                             kernel_size=7,
                             stride=1,
                             padding=7 >> 1,
                             bias=True)
      self.conv2 = nn.Conv2d(5, 5, 5, 1, 5 >> 1, bias=True)
      self.conv3 = nn.Conv2d(5, 10, 3, 1, 3 >> 1, bias=True)
      self.conv4 = nn.Conv2d(10, 1, 1, 1, 1 >> 1, bias=True)

    def forward(self, x):
        F0 = self.condition_conv(x)
        coarse_transMap = torch.sigmoid(F0)
        F_ = self.conv1(x)
        F = torch.cat((F_, coarse_transMap), 1)
        F_0 = self.conv2(F)
        F_1 = self.conv3(F_0)
        fine_transMap = torch.sigmoid(self.conv4(F_1))
        fine_transMap = torch.cat([fine_transMap]*3, 1)

        return fine_transMap

class AFA(nn.Module):
    def __init__(self, m=-0.68):
        super(AFA, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class SFT_layer(nn.Module):
    def __init__(self):
        super(SFT_layer, self).__init__()
        Relu = nn.LeakyReLU(0.2, True)

        condition_conv1 = nn.Conv2d(1,16, kernel_size=3, stride=1, padding=1)
        condition_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        condition_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        conditon_conv = [condition_conv1, Relu, condition_conv2, Relu, condition_conv3, Relu]
        self.condition_conv = nn.Sequential(*conditon_conv)

        scale_conv1 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)
        scale_conv2 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)
        scale_conv = [scale_conv1, Relu, scale_conv2, Relu]
        self.scale_conv = nn.Sequential(*scale_conv)

        sift_conv1 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)
        sift_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        sift_conv = [sift_conv1, Relu, sift_conv2, Relu]
        self.sift_conv = nn.Sequential(*sift_conv)

    def forward(self, x, depth):
        depth_condition = self.condition_conv(depth)
        scaled_feature = self.scale_conv(depth_condition) * x
        sifted_feature = scaled_feature + self.sift_conv(depth_condition)

        return sifted_feature


class Haze_transferNet(nn.Module):
    def __init__(self):
      super(Haze_transferNet, self).__init__()

      self.Trans_fineNet = Trans_fineNet()

      for param in self.Trans_fineNet.parameters():
          param.requires_grad = True

      Relu = nn.LeakyReLU(0.2, True)

      condition_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
      condition_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
      condition_conv3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

      conditon_conv = [condition_conv1, Relu, condition_conv2, Relu, condition_conv3, Relu]
      self.condition_conv02 = nn.Sequential(*conditon_conv)

      scale_conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
      scale_conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
      scale_conv = [scale_conv1, Relu, scale_conv2, Relu]
      self.scale_conv = nn.Sequential(*scale_conv)

      sift_conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
      sift_conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
      sift_conv = [sift_conv1, Relu, sift_conv2, Relu]
      self.sift_conv = nn.Sequential(*sift_conv)
      self.MIN_BETA = 0.010
      self.MAX_BETA = 0.035

    def forward(self, hazy_img, clear_img, depth):
        # F
        fine_betaMap = self.Trans_fineNet(hazy_img)

        depth_condition = self.scale_conv(depth)
        fine_transMap = depth_condition * fine_betaMap
        fine_transMap = fine_transMap + self.sift_conv(depth_condition)


        a = random.random()
        fine_transMap = fine_transMap * depth * a
        fine_transMap = ((torch.tanh(fine_transMap) + 1) / 2)
        fine_transMap = fine_transMap.clamp(0.05, 0.95)
        airlight = atmospheric_light(clear_img)

        fake_hazy = clear_img * fine_transMap + (1 - fine_transMap) * airlight

        return fake_hazy




if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    clear_img = torch.rand((1, 3, 256, 256))
    depth = torch.rand((1, 3, 256, 256))
    x = x.type(torch.FloatTensor)
    clear_img = clear_img.type(torch.FloatTensor)
    depth = depth.type(torch.FloatTensor)
    out= Haze_transferNet(x, depth, clear_img)
    print(out)
