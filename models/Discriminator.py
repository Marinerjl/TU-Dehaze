import torch
import torch.nn as nn

class _FeatureBlockDiscriminator2(nn.Module):
    def __init__(self, input_nc=128):
        super(_FeatureBlockDiscriminator2, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Linear(540800, input_nc),
            nn.PReLU(),
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(input_nc, input_nc),
            nn.PReLU(),
            nn.Sigmoid()
        )

        self.layer_3 = nn.Linear(input_nc, 2)

    def forward(self, x2):
        result2 = x2.contiguous().view(-1, x2.size(1) * x2.size(2) * x2.size(3))  # 306432
        layer1 = self.layer_1(result2)
        layer2 = self.layer_2(layer1)
        output = self.layer_3(layer2)
        return output


class _FeatureBlockDiscriminator4(nn.Module):
    def __init__(self, input_nc=128):
        super(_FeatureBlockDiscriminator4, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Linear(147968, input_nc),
            nn.PReLU(),
            nn.Sigmoid()
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(input_nc, input_nc),
            nn.PReLU(),

        )

        self.layer_3 = nn.Linear(input_nc, 2)

    def forward(self, x4):
        result4 = x4.contiguous().view(-1, x4.size(1) * x4.size(2) * x4.size(3))  # 306432
        layer1 = self.layer_1(result4)
        layer2 = self.layer_2(layer1)
        output = self.layer_3(layer2)

        return output

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs

if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    net = Discriminator(input_nc=3,ndf=64, norm_layer=nn.BatchNorm2d)
    out = net(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    net2 = _FeatureBlockDiscriminator4(3)
    out2 = net2(out[1])
    print(out2.shape)
    # print(net)