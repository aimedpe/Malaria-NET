import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from typing import List

def load_modelFastRCNN(model_ckpt: str, device: str) -> torch.nn.Module:
    """
    Load a Faster R-CNN model with a custom number of classes and pre-trained weights.

    Parameters:
    -----------
    model_ckpt (str): File path to the model checkpoint containing the pre-trained weights.
    device (str): Device to which the model will be moved, e.g., 'cpu' or 'cuda'.

    Returns:
    -------
    model (torch.nn.Module): Loaded Faster R-CNN model with custom number of classes and weights.

    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    try: 
        model.to(device)
        state_dict = torch.load(model_ckpt, map_location=device)['model_ckpt']
        model.load_state_dict(state_dict)
    except:
        model.to(device)
        state_dict = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(state_dict)


    return model


def get_bboxes_output(output: List[dict], eps: float) -> torch.Tensor:
    """
    Get bounding box predictions from the model's output and perform Non-Maximum Suppression (NMS).

    Parameters:
    -----------
    output (List[dict]): Model's output containing bounding box predictions and scores for each class.
    eps (float): The threshold value for NMS. Boxes with IoU (Intersection over Union) higher than
                 this value will be suppressed.

    Returns:
    -------
    keep (torch.Tensor): Indexes of the kept bounding boxes after performing NMS.

    """
    out_bbox = output[0]["boxes"]
    out_scores = output[0]["scores"]
    keep = torchvision.ops.nms(out_bbox, out_scores, eps)
    return keep