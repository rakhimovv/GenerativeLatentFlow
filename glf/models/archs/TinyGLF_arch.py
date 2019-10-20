from torch import nn as nn

from models.archs.arch_util import initialize_weights


class TinyEncoder(nn.Module):
    def __init__(self, img_size, in_ch, nz):
        super().__init__()

        M = img_size // 4

        # We use kernel size 3x3 instead of 4x4 as claimed by the original authors
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * M * M, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, nz)
        )

        initialize_weights(self)

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out)
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
