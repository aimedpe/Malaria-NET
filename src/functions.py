import torch
from torchvision.transforms import Compose,Normalize,ToTensor

import numpy as np
from PIL import Image


import src.process_image as PI
from src.resnet import load_resnet50
from src.fastrcnn import load_modelFastRCNN,get_bboxes_output



def return_predictionsFastrcnn(model,image,device):

    model.eval()
    
    transform = Compose([
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406],
              std = [0.229, 0.224, 0.225])
    ])

    image = image.resize((2560,2560),resample = Image.Resampling.BICUBIC)

    image_np = np.array(image)

    image_tensor = transform(image).to(device)

    output = model([image_tensor])

    idxs = get_bboxes_output(output,0.99)

    bboxes = []

    with torch.no_grad():

        for idx in idxs:

            bbox = output[0]["boxes"][idx].cpu().numpy()

            #if not check_bounding_boxes(bbox,bboxes):
            
            bboxes.append(bbox)

    bboxes = PI.get_processed_bboxes(bboxes,(2560,2560))

    del model
    
    torch.cuda.empty_cache()

    return bboxes,image_np




def predictionFastrcnn(model_ckpt:str, k_image:str):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_modelFastRCNN(model_ckpt,device)

    image_pil = Image.open(k_image)

    image_np = np.array(image_pil)

    image_np = PI.crop_sample_image(image_np,100)

    image_pil = Image.fromarray(image_np)

    bboxes,image_np = return_predictionsFastrcnn(model,image_pil,device)
    
    cell_images = PI.get_cell_images_fastrcnn(image_np,bboxes)

    return cell_images,bboxes




def predictionResNet(model_ckpt:str ,n_classes: int,n_images: list[np.ndarray],n_bboxes:list[np.array]):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_resnet50(n_classes,False,model_ckpt,device)

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
            #POSIBLE ERROR DE PREPROCESAMIENTO
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

def process_predictionsResNet(predictions: list[np.ndarray]):

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


def process_results(results_falciparum,results_vivax,results_duplicates,thres_fal=5,thres_viv=2):

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

def process_probs_to_results(probabilities: list[np.ndarray]):

    results_falciparum = {"n_infected":0}
    results_vivax = {"n_infected":0}

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

    if len(probabilities)!=0:

        return total_f/len(probabilities), total_v/len(probabilities), results_falciparum, results_vivax

    else:

        return 0,0,results_falciparum,results_vivax

def process_probsResNet(probabilities: list[np.ndarray]):

    total_f = 0
    total_v = 0

    for p in probabilities:

        prob_f, prob_v = p[0]
        #if prob_f > prob_v: total_f += 1
        #else: total_v += 1
        total_f += prob_f
        total_v += prob_v
    
    return total_f/len(probabilities), total_v/len(probabilities)


def process_probsResNet_duplicates(probabilities: list[np.ndarray]):

    results = []

    for p in probabilities:

        prob_f, prob_v = p[0]
        if prob_f > prob_v:  results.append(0)
        else: results.append(1)
    
    return results



def filtering_duplicates(falciparum_bboxes,candidates_falciparum,vivax_bboxes,candidates_vivax):

    duplicate_images,duplicate_bboxes = PI.find_duplicate_bounding_boxes(falciparum_bboxes,
                                                                         candidates_falciparum,
                                                                         vivax_bboxes,candidates_vivax)
    
    return duplicate_images,duplicate_bboxes

