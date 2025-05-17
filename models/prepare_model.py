# This script prepares the frozen DINO ViT-S/16 model for transfer learning.
# It loads the model from Facebook's DINO repository, freezes its parameters, and adds a classification head.

import torch
import torch.nn as nn

def get_frozen_dino_vits16_model(device, num_classes=100):
    base_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    # Freeze all base model parameters
    for param in base_model.parameters():
        param.requires_grad = False # freeza parametri 

    # Compose full model with additional classification head
    class DinoClassifier(nn.Module):
        def __init__(self, base, num_classes):
            super().__init__()
            self.base = base
            self.classifier = nn.Linear(base.num_features, num_classes)

        def forward(self, x):
            x = self.base(x)
            return self.classifier(x)

    model = DinoClassifier(base_model, num_classes)
    return model.to(device)