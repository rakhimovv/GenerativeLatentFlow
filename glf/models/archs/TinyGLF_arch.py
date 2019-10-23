from torch import nn as nn

from glf.models.archs.arch_util import initialize_weights


class MeanAggregator(nn.Module):
    def forward(self, x):
        return x.mean(dim=1, keepdim=True)


class Flattener(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Reshaper(nn.Module):
    def __init__(self, new_shape):
        super().__init__()
        assert isinstance(new_shape, list)
        self.new_shape = new_shape

    def forward(self, x):
        return x.reshape([-1] + self.new_shape)


class TinyEncoder(nn.Module):
    def __init__(self, img_size, in_ch, nz):
        super().__init__()

        assert img_size == 64, 'TinyEncoder works only for [CH, 64, 64] images'

        self.conv = nn.Sequential(
                MeanAggregator(),
                nn.BatchNorm2d(1, affine=False),
                nn.Conv2d(1, 8, 5, 3),
                nn.LeakyReLU(),
                nn.Conv2d(8, 16, 3, 2),
                nn.LeakyReLU(),
                nn.Conv2d(16, 32, 2, 1),
                nn.LeakyReLU(),
                Flattener(),
                nn.Linear(2048, nz)
        )

        # initialize_weights(self)

    def forward(self, x):
        out = self.conv(x)
        return out


class TinyDecoder(nn.Module):
    def __init__(self, img_size, out_ch, nz):
        super().__init__()
        M = img_size // 32
        self.M = M

        self.fc = nn.Sequential(
            nn.Linear(nz, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 8 * M * M, bias=False),
            nn.BatchNorm1d(8 * M * M),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, 8, self.M, self.M)
        out = self.deconv(out)
        return out
