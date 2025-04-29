import torch.nn as nn
import torch.nn.functional as F

class EllipseCounterCNN(nn.Module):
    def __init__(self):
        super(EllipseCounterCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 512x512 → 512x512
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # 512x512 → 256x256
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 256x256 → 256x256
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # 256x256 → 128x128
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 128x128 → 128x128
            nn.ReLU(),
            nn.MaxPool2d(2)                                       # 128x128 → 64x64
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
           )   # Regression output (ellipse count)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x