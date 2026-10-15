"""Five residual blocks and five max-pools as in Fig. 1 and Section IV-A."""
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.shortcut = nn.Conv2d(cin, cout, 1, bias=False)
        self.pool = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        return self.pool(self.main(x) + self.shortcut(x))


class CNNEncoder(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        widths = [3, dim, dim * 2, dim * 4, dim * 4, dim * 8]
        self.blocks = nn.Sequential(*[ResidualBlock(a, b) for a, b in zip(widths, widths[1:])])
        self.out_channels = widths[-1]

    def forward(self, image):
        return self.blocks(image)
