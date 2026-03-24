import torch
import torch.nn as nn

# [TEMPLATE] Set this to the winning configuration found by the Binary Architecture Search!
WINNING_WIDTHS = [8, 16, 32] 

class DynamicNet(nn.Module):
    """
    Dynamically constructs a Convolutional Neural Network.
    The depth and width of the network are controlled by the `widths` list.
    """
    def __init__(self, widths=WINNING_WIDTHS):
        super().__init__()
        assert len(widths) >= 2, "widths list must contain at least 2 layer sizes"
        
        # Initial Convolutional Block
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dynamic Feature Construction
        layers = []
        in_ch = widths[0]
        
        for w in widths[1:]:
            layers.append(nn.Conv2d(in_ch, w, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(w))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            in_ch = w
            
        self.features = nn.Sequential(*layers)
        
        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.features(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
