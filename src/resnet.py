import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50


def load_resnet50(num_classes: int, pretrained: bool = False, ckpt: str = None, device: str = 'cpu') -> torch.nn.Module:
    """
    Load a ResNet-50 model with a custom number of output classes and pre-trained weights.

    Parameters:
    -----------
    num_classes (int): Number of output classes for the custom classifier.
    pretrained (bool, optional): If True, load the model with pre-trained weights. Default is False.
    ckpt (str, optional): File path to the model checkpoint containing the pre-trained weights.
                          If provided, the model will be loaded from the checkpoint. Default is None.
    device (str, optional): Device to which the model will be moved, e.g., 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
    -------
    torch.nn.Module: Loaded ResNet-50 model with a custom classifier and pre-trained weights (if applicable).

    """
 
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

