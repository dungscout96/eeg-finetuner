import inspect
from torch import nn
from braindecode.models import Labram

class LaBRAMBackbone(nn.Module):
    def __init__(self, 
            **kwargs
        ):
        super(LaBRAMBackbone, self).__init__()
        
        print(f"Initializing Labram with kwargs: {kwargs}")
        self.network = Labram(**kwargs)

    def forward(self, x):
        return self.network(x)