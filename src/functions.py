import torch
from torchvision.transforms import Compose,Normalize,ToTensor

import numpy as np
from PIL import Image
from typing import List,Dict,Tuple

import src.process_image as PI
from src.resnet import load_resnet50
from src.fastrcnn import load_modelFastRCNN,get_bboxes_output



def return_predictionsFastrcnn(model: torch.nn.Module, image: Image.Image, device: str) -> Tuple[List[np.ndarray],np.ndarray]:
    """
    Perform inference using a Faster R-CNN model to detect objects in an image.

    Parameters:
    -----------
    model (torch.nn.Module): Preloaded and configured Faster R-CNN model.
    image (PIL.Image.Image): Input image for object detection.
    device (str): Device on which the inference will be performed, e.g., 'cpu' or 'cuda'.

    Returns:
    -------
    bboxes (list): List of detected bounding boxes, each represented as a Numpy array.
    image_np (numpy.ndarray): Numpy representation of the input image.

    """
    model.eval()

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = image.resize((2560, 2560), resample=Image.Resampling.BICUBIC)

    image_np = np.array(image)

    image_tensor = transform(image).to(device)

    output = model([image_tensor])

    idxs = get_bboxes_output(output, 0.99)

    bboxes = []

    with torch.no_grad():

        for idx in idxs:

            bbox = output[0]["boxes"][idx].cpu().numpy()

            bboxes.append(bbox)

    bboxes = PI.get_processed_bboxes(bboxes, (2560, 2560))

    del model

    torch.cuda.empty_cache()

    return bboxes, image_np




def predictionFastrcnn(model_ckpt:str, k_image:str) -> Tuple[List[np.ndarray],List[np.ndarray]]:

    """
    Perform object detection using a Faster R-CNN model on an image.

    Parameters:
    -----------
    model_ckpt (str): File path to the model checkpoint containing pre-trained weights.
    k_image (str): File path to the input image for object detection.

    Returns:
    -------
    cell_images (list): List of cropped cell images detected by Faster R-CNN.
    bboxes (list): List of bounding boxes around the detected objects, each represented as a Numpy array.

    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_modelFastRCNN(model_ckpt,device)

    image_pil = Image.open(k_image)

    image_np = np.array(image_pil)

    image_np = PI.crop_sample_image(image_np,100)

    image_pil = Image.fromarray(image_np)

    bboxes,image_np = return_predictionsFastrcnn(model,image_pil,device)
    
    cell_images = PI.get_cell_images_fastrcnn(image_np,bboxes)

    return cell_images,bboxes




def predictionResNet(model_ckpt: str, 
                     n_classes: int, 
                     n_images: List[np.ndarray], 
                     n_bboxes: List[np.ndarray])->Tuple[List[np.ndarray],List[np.ndarray],List[np.ndarray]]:
    """
    Perform image classification using a ResNet-50 model on a list of images.

    Parameters:
    -----------
    model_ckpt (str): File path to the model checkpoint containing pre-trained weights.
    n_classes (int): Number of classes in the classification problem.
    n_images (List[np.ndarray]): List of NumPy arrays representing input images for classification.
    n_bboxes (List[np.array]): List of bounding boxes for the corresponding images.

    Returns:
    -------
    predictions (List[np.ndarray]): List of predicted probabilities for each class for each image.
    candidates (List[np.ndarray]): List of images classified as an infected.
    new_bboxes (List[np.ndarray]): List of bounding boxes for the images classified as an infected.
    """


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_resnet50(n_classes,model_ckpt,device)

    predictions = []
    candidates = []
    new_bboxes = []

    transform = Compose([
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406],
                  std = [0.229, 0.224, 0.225])
    ])
    model.eval()

    image_size = (128,128)

    for i,image in enumerate(n_images):
        bbox = n_bboxes[i]
        try:
            image_pil = Image.fromarray(image).resize(image_size,resample=Image.Resampling.BICUBIC)
            image_tensor = transform(image_pil)
            pred = model(image_tensor.unsqueeze(0).to(device))
            pred = torch.softmax(pred.data,1)

            if pred.cpu().numpy().argmax() == 1:
                candidates.append(image)    
                new_bboxes.append(bbox)

            predictions.append(pred.cpu().numpy())
        except:
            pass

    return (predictions,candidates,new_bboxes)

def process_predictionsResNet(predictions: list[np.ndarray])->Dict:

    """
    Process the predictions from the ResNet-50 model for image classification.

    Parameters:
    -----------
    predictions (List[np.ndarray]): List of predicted probabilities for each class for each image.

    Returns:
    -------
    results (Dict): Dictionary containing processed information from the predictions.
                    - 'probabilities' (List): List of probabilities for the class at index 1 (infected).
                    - 'n_infected' (int): Number of images classified as infected.

    """

    n_infected = 0
    prob = []
    pred = []
    
    for p in predictions:
        prob.append(p[0])
        pred.append(p[0].argmax())
        if p[0].argmax() == 1: n_infected += 1

    results = {
        'probabilities': prob,
        'n_infected': n_infected
    }

    return results


def process_results(results_falciparum: Dict, 
                    results_vivax: Dict, 
                    results_duplicates: List, 
                    thres_fal: int = 5, 
                    thres_viv: int = 2)-> Tuple[int,int]:
    
    """
    Process the results from multiple classifications for Plasmodium falciparum and Plasmodium vivax.

    Parameters:
    -----------
    results_falciparum (dict): Results from the classification of Plasmodium falciparum.
    results_vivax (dict): Results from the classification of Plasmodium vivax.
    results_duplicates (list): List of classification results for duplicate images.
    thres_fal (int): Threshold for considering a positive classification for Plasmodium falciparum.
    thres_viv (int): Threshold for considering a positive classification for Plasmodium vivax.

    Returns:
    -------
    pf (int): Binary indicator (0 or 1) whether Plasmodium falciparum is detected or not.
    pv (int): Binary indicator (0 or 1) whether Plasmodium vivax is detected or not.

    """


    pv = 0
    pf = 0

    size_duplicates = len(results_duplicates)

    results_falciparum['n_infected'] -= size_duplicates

    results_vivax['n_infected'] -= size_duplicates

    falciparum_dup = np.count_nonzero(np.array(results_duplicates)==0)
    vivax_dup = np.count_nonzero(np.array(results_duplicates)==1)

    results_falciparum['n_infected'] += falciparum_dup
    results_vivax['n_infected'] += vivax_dup

    if results_falciparum['n_infected']>thres_fal: pf = 1
    if results_vivax['n_infected']>thres_viv: pv = 1

    return pf,pv

def process_probs_to_results(probabilities: List[np.ndarray]) -> Tuple[float,float,Dict,Dict]:
    """
    Process probabilities to determine results for Plasmodium falciparum and Plasmodium vivax.

    Parameters:
    -----------
    probabilities (List[np.ndarray]): List of predicted probabilities for each class for each image.

    Returns:
    -------
    f_ratio (float): Ratio of images classified as Plasmodium falciparum to total images.
    v_ratio (float): Ratio of images classified as Plasmodium vivax to total images.
    results_falciparum (dict): Results from the classification of Plasmodium falciparum.
    results_vivax (dict): Results from the classification of Plasmodium vivax.

    """
    results_falciparum = {"n_infected": 0}
    results_vivax = {"n_infected": 0}

    total_f = 0
    total_v = 0

    for p in probabilities:
        prob_f, prob_v = p[0]

        if prob_f > prob_v:
            results_falciparum["n_infected"] += 1
            total_f += 1
        else:
            results_vivax["n_infected"] += 1
            total_v += 1

    if len(probabilities) != 0:
        f_ratio = total_f / len(probabilities)
        v_ratio = total_v / len(probabilities)
        return f_ratio, v_ratio, results_falciparum, results_vivax
    else:
        return 0, 0, results_falciparum, results_vivax


def process_probsResNet(probabilities: List[np.ndarray]) -> Tuple[float,float]:
    """
    Process probabilities to determine the average probabilities for Plasmodium falciparum and Plasmodium vivax.

    Parameters:
    -----------
    probabilities (List[np.ndarray]): List of predicted probabilities for each class for each image.

    Returns:
    -------
    avg_prob_f (float): Average probability for Plasmodium falciparum.
    avg_prob_v (float): Average probability for Plasmodium vivax.

    """
    total_f = 0
    total_v = 0

    for p in probabilities:
        prob_f, prob_v = p[0]
        total_f += prob_f
        total_v += prob_v

    if len(probabilities) != 0:
        avg_prob_f = total_f / len(probabilities)
        avg_prob_v = total_v / len(probabilities)
    else:
        avg_prob_f = 0
        avg_prob_v = 0

    return avg_prob_f, avg_prob_v


def process_probsResNet_duplicates(probabilities: List[np.ndarray]):
    """
    Process probabilities to determine the results for duplicates based on Plasmodium falciparum and Plasmodium vivax.

    Parameters:
    -----------
    probabilities (List[np.ndarray]): List of predicted probabilities for each class for each image.

    Returns:
    -------
    results (List[int]): List of binary indicators (0 or 1) for each image where 0 indicates Plasmodium falciparum
                         and 1 indicates Plasmodium vivax.

    """
    results = []

    for p in probabilities:
        prob_f, prob_v = p[0]
        if prob_f > prob_v:
            results.append(0)
        else:
            results.append(1)

    return results


def filtering_duplicates(falciparum_bboxes: List[np.ndarray], 
                         candidates_falciparum: List[np.ndarray], 
                         vivax_bboxes: List[np.ndarray], 
                         candidates_vivax:List[np.ndarray]):
    """
    Filter and find duplicate images and bounding boxes between Plasmodium falciparum and Plasmodium vivax.

    Parameters:
    -----------
    falciparum_bboxes (list): List of bounding boxes for Plasmodium falciparum images.
    candidates_falciparum (list): List of candidate images for Plasmodium falciparum.
    vivax_bboxes (list): List of bounding boxes for Plasmodium vivax images.
    candidates_vivax (list): List of candidate images for Plasmodium vivax.

    Returns:
    -------
    duplicate_images (list): List of duplicate images found between Plasmodium falciparum and Plasmodium vivax.
    duplicate_bboxes (list): List of bounding boxes corresponding to the duplicate images.

    """
    duplicate_images, duplicate_bboxes = PI.find_duplicate_bounding_boxes(falciparum_bboxes,
                                                                          candidates_falciparum,
                                                                          vivax_bboxes,
                                                                          candidates_vivax)

    return duplicate_images, duplicate_bboxes
