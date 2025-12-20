from torch import nn

class LaBRAMBackbone(nn.Module):
    def __init__(self, 
            input_size, 
            num_channels,
            embedding_size=128,
            hidden_sizes=[256, 128],
            dropout_rate=0.5
        ):
        super(LaBRAMBackbone, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, embedding_size))
        layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)