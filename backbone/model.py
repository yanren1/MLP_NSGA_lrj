import torch.nn as nn
import torchvision

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Linear(8, 3),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    backbone = torchvision.ops.MLP(in_channels=27,
                                # hidden_channels=[28, 32, 64, 128, 256, 128, 64, 32, 16, 8, 3],
                                hidden_channels=[28, 64, 256, 64, 8, 3],

                                # norm_layer=nn.LayerNorm,
                                dropout= 0,inplace=False).cuda()