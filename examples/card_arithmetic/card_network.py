"""The card classifier used by the original card arithmetic examples."""

import torch


class CardEncoder(torch.nn.Module):
    """The feature layers of the original card classifier: four convolution and
    pooling layers and a linear layer, from a card image to 512 features.
    """

    def __init__(self) -> None:
        """The layers of the original card examples, up to the 512 features."""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
        self.conv2 = torch.nn.Conv2d(64, 64, 3)
        self.conv3 = torch.nn.Conv2d(64, 128, 3)
        self.conv4 = torch.nn.Conv2d(128, 128, 3)
        self.fc1 = torch.nn.Linear(17280, 512)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.5)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512 features of each card image, (images, 3, 274, 174)."""
        x = self.pool(self.ReLU(self.conv1(x)))
        x = self.pool(self.ReLU(self.conv2(x)))
        x = self.pool(self.ReLU(self.conv3(x)))
        x = self.pool(self.ReLU(self.conv4(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc1(x)


class CardNet(torch.nn.Module):
    """The card classifier of the original card arithmetic examples: 52 classes per
    image.
    """

    def __init__(self) -> None:
        """CardEncoder followed by a 52-class linear layer, as in the original. The
        layers are created in the original order, so a seed gives the same weights.
        """
        super().__init__()
        self.encoder = CardEncoder()
        self.fc2 = torch.nn.Linear(512, 52)
        self.softmax = torch.nn.Softmax(1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512 features of each card image, (images, 3, 274, 174), before
        the class layer.
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the probabilities of the 52 classes for each card image."""
        # Input: card images, (images, 3, 274, 174); output: (images, 52) probabilities.
        return self.softmax(self.fc2(self.features(x)))


class DirectCardNet(torch.nn.Module):
    """A purely neural baseline: predict the result of a tuple of cards directly,
    without the card classes or the program.

    Each card goes through a CardEncoder, CardNet's layers up to its 512 features; the
    features of the tuple's cards, in order, are concatenated and a linear layer
    scores every possible result.
    """

    def __init__(self, cards: int, outputs: int) -> None:
        """A network for tuples of cards cards with outputs possible results."""
        super().__init__()
        self.cards = cards
        self.encoder = CardEncoder()
        self.classifier = torch.nn.Linear(cards * 512, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits of the results for card tuples, (batch, cards, 3, 274,
        174) -> (batch, outputs).
        """
        # All cards of all tuples in one call, tuple by tuple: (batch * cards, 512).
        features = self.encoder(x.flatten(0, 1))
        return self.classifier(features.reshape(len(x), self.cards * 512))
