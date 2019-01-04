import torch
import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self, act='relu'):
        super().__init__()

        self.features = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                self.activation(act),
                nn.AvgPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                self.activation(act),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )

        self.fc = nn.Sequential(
                nn.Linear(in_features=4 * 4 * 16, out_features=120),
                self.activation(act),
                nn.Linear(in_features=120, out_features=84),
                self.activation(act),
                nn.Linear(in_features=84, out_features=10)
            )

    def forward(self, x):
        assert x.dim() == 4 and x.size()[1:] == (1, 28, 28)

        features = self.features(x)
        features = features.view(-1, 4 * 4 * 16)
        out = self.fc(features)

        return out

    @staticmethod
    def activation(act):
        assert act in ['relu', 'sigmoid', 'tanh']

        if act == 'relu':
            return nn.ReLU(True)
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'tanh':
            return nn.Tanh()


if __name__ == '__main__':
    model = LeNet()
    print(model)

    x = torch.zeros(5, 1, 28, 28)
    out = model(x)

    print(out.size())
