import torch
import torch.nn as nn


class Net(nn.Module):
    """The digit classifier of the original NeurASP MNIST examples: 10 classes per
    image.
    """

    def __init__(self) -> None:
        """A LeNet-style digit classifier: two convolution layers and three linear
        layers.
        """
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 6, 5
            ),  # 6 is the output chanel size; 5 is the kernal size; 1 (chanel) 28 28 -> 6 24 24
            nn.MaxPool2d(2, 2),  # kernal size 2; stride size 2; 6 24 24 -> 6 12 12
            nn.ReLU(
                True
            ),  # inplace=True means that it will modify the input directly thus save memory
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Softmax(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the probabilities of the 10 digits for a (batch, 1, 28, 28)."""
        x = self.encoder(x)
        x = x.flatten(1)  # reshape, not view: works in the channels_last layout too
        x = self.classifier(x)
        return x
