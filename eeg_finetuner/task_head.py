from torch import nn

def get_task_head(task_type: str, decoder_type: str, **kwargs):
    if task_type == "classification":
        return get_classifier(decoder_type, **kwargs)
    elif task_type == "regression":
        return get_regressor(decoder_type, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def get_classifier(classifier_type: str, **kwargs):
    if classifier_type == "linear":
        return LinearHead(**kwargs)
    elif classifier_type == "mlp":
        return MLPHead(**kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
def get_regressor(regressor_type: str, **kwargs):
    if regressor_type == "linear":
        return LinearHead(num_classes=1, **kwargs)
    elif regressor_type == "mlp":
        return MLPHead(num_classes=1, **kwargs)
    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}")

class LinearHead(nn.Module):
    def __init__(self, 
            input_size, 
            num_classes,
            embedding_size=128,
            dropout_rate=0.5
        ):
        super(LinearHead, self).__init__()
        self.layer1 = nn.Linear(input_size, embedding_size)
        self.output_layer = nn.Linear(embedding_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

class MLPHead(nn.Module):
    def __init__(self, 
            input_size, 
            num_classes,
            hidden_sizes=[256, 128],
            dropout_rate=0.5
        ):
        super(MLPHead, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)