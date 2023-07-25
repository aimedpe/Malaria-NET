import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50


def load_resnet50(num_classes:int ,pretrained=False,ckpt: str = None,device: str = 'cpu'):
    
    if pretrained: model = resnet50(weights = ResNet50_Weights)
    else: model = resnet50(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)

    if ckpt is None:
        model.to(device)
    else:
        state_dict = torch.load(ckpt,map_location="cuda:0")["model_ckpt"]
        model.load_state_dict(state_dict)
        model.to(device)

    return model

