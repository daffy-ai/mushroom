import torch.nn as nn
from torchvision.models import resnet18

def get_model(num_classes=2):
    model = resnet18(weights='IMAGENET1K_V1')  # pretrained=True deprecated
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model