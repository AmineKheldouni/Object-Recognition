import numpy as np
from torch import nn
from torchvision import models


class FineTuneModel(nn.Module):
    """Model used to test a ResNet50 with finetuning FC layers"""
    def __init__(self, num_classes):
        super(FineTuneModel, self).__init__()
        # Everything except the last linear layer
        original_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
