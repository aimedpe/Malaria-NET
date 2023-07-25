from PIL import Image,ImageDraw
import numpy as np
from skimage.measure import label, regionprops
from typing import List
import cv2



def crop_sample_image(image: np.ndarray, alpha: int) -> np.ndarray:
    
    '''
    Parameters:
    -----------

    image: Toma la imagen de la muestra de sangre sin procesar; es decir, con todos los bordes.
    alpha: Parámetro para crear una máscara que mostrará el borde de la muestra.

    Returns:
    --------

    new_image: Es la muestra recortada; es decir, sin bordes oscuros. Basicamente un circulo inscrito en un cuadrado.
    
    '''

    mask = (image[:,:,0]>alpha).astype('uint8')
    mask = np.array(mask)
    obj_ids = np.unique(mask)
    
    obj_ids = obj_ids[1:]
    
    masks = mask == obj_ids[:, None, None]
    
    num_objs = len(obj_ids)
    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

    assert len(boxes)==1, "Error getting the mask of the main image"

    new_image = image[ymin:ymax,xmin:xmax,:]

    return new_image
 

def draw_bounding_boxes(image:np.ndarray, bboxes:list[tuple]) -> Image.Image:

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for bbox in bboxes:
        min_row, min_col, max_row, max_col = bbox
        xmin,ymin,xmax,ymax = bbox  
        draw.rectangle((xmin, ymin, xmax, ymax), outline="green", width=7)
    output_image = np.array(pil_image)
    output_image = Image.fromarray(output_image)

    return output_image


def get_cell_images_fastrcnn(image: np.ndarray,bboxes: list) -> List[np.ndarray]:

   cells = []

   for b in bboxes:
        xmin,ymin,xmax,ymax = b  
        cell = image[int(ymin):int(ymax),int(xmin):int(xmax),:]
        cells.append(cell)

   return cells


def get_bounding_boxes_fastrcnn(mask:np.ndarray) -> List[tuple]:
    
    labeled = label(mask)
    props = regionprops(labeled)
    bboxes = []
    for prop in props:
        ymin,xmin,ymax,xmax = prop.bbox
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes

def get_processed_bboxes(bboxes,size_image):

    mask = np.zeros(size_image)#en formato numpy el size

    for bbox in bboxes:

        xmin,ymin,xmax,ymax = bbox

        top_left = (int(xmin),int(ymax))
        bottom_right = (int(xmax),int(ymin))

        cv2.rectangle(mask,top_left,bottom_right,255,-1)
    
    bboxes_processed = get_bounding_boxes_fastrcnn(mask)

    return bboxes_processed


def calculate_iou(box_a, box_b):
    # Coordenadas de la intersección
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    # Área de la intersección
    intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

    # Área de las bounding boxes
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Índice de intersección sobre unión (IoU)
    iou = intersection_area / float(area_a + area_b - intersection_area)

    return iou

def find_duplicate_bounding_boxes(bounding_boxes1,candidates1,bounding_boxes2,candidates2,alpha=0.5):

    duplicate_boxes = []
    duplicate_images = []
    for i,box1 in enumerate(bounding_boxes1):
        for j,box2 in enumerate(bounding_boxes2):
            iou = calculate_iou(box1, box2)
            if iou > alpha:
                # Determinar el bounding box con mayor área
                area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                if area_box1 >= area_box2:
                    duplicate_boxes.append(box1)
                    duplicate_images.append(candidates1[i])
                else:
                    duplicate_boxes.append(box2)
                    duplicate_images.append(candidates2[j])
    return duplicate_images,duplicate_boxes
