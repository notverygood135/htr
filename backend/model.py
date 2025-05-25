from torch import nn
import torch

class _BidirectionalLSTM(nn.Module):
    def __init__(self, inputs_size: int, hidden_size: int, output_size: int):
        super(_BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(inputs_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.lstm(x)
        sequence_length, batch_size, inputs_size = recurrent.size()
        sequence_length2 = recurrent.view(sequence_length * batch_size, inputs_size)

        out = self.linear(sequence_length2)  # [sequence_length * batch_size, output_size]
        out = out.view(sequence_length, batch_size, -1)  # [sequence_length, batch_size, output_size]

        return out

class CRNN(nn.Module):
    def __init__(self, num_classes: int, input_channels=1):
        super(CRNN, self).__init__()

        # original image size: 58x1068
        # CNN Layers (reduce height to 1)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=True),  # (B, 64, 58, 1068)
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),  # (B, 64, 29, 534)

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),  # (B, 128, 14, 267)

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),  # (B, 256, 7, 133)

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (B, 512, 3, 133)

            nn.Conv2d(512, 512, (3, 3), (1, 1), (0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # (B, 512, 1, 133)
        )

        self.rnn = nn.Sequential(
            _BidirectionalLSTM(512, 256, 256),
            _BidirectionalLSTM(256, 256, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)  # shape: (B, C, H, W)
        b, c, h, w = x.size()
        assert h == 1, "height must be 1 after CNN"
        x = x.squeeze(2)        # (B, C, W)
        x = x.permute(2, 0, 1)  # (W, B, C)

        x = self.rnn(x)      # (W, B, num_classes)
        return x