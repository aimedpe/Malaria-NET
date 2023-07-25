from PIL import Image,ImageDraw
import numpy as np
from skimage.measure import label, regionprops
from typing import List,Tuple
import cv2



def crop_sample_image(image: np.ndarray, alpha: int) -> np.ndarray:

    """
    Crop the main object from the input image based on a specified alpha value.

    Parameters:
    -----------
    image (np.ndarray): Input image represented as a NumPy array.
    alpha (int): Threshold value for alpha (transparency) channel of the image.

    Returns:
    -------
    np.ndarray: Cropped image containing the main object based on the specified alpha threshold.

    Raises:
    ------
    AssertionError: If more than one object is found in the image or no object is found.

    """

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
 
def draw_bounding_boxes(image: np.ndarray, bboxes: List[Tuple]) -> Image.Image:
    """
    Draw bounding boxes on an image.

    Parameters:
    -----------
    image (np.ndarray): Input image represented as a NumPy array.
    bboxes (list of tuples): List of bounding box tuples (xmin, ymin, xmax, ymax)
                             specifying the coordinates of the bounding boxes.

    Returns:
    -------
    Image.Image: An image with bounding boxes drawn around the specified regions.

    """

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for bbox in bboxes:
        xmin,ymin,xmax,ymax = bbox  
        draw.rectangle((xmin, ymin, xmax, ymax), outline="green", width=7)
    output_image = np.array(pil_image)
    output_image = Image.fromarray(output_image)

    return output_image


def get_cell_images_fastrcnn(image: np.ndarray, bboxes: List[np.ndarray]) -> List[np.ndarray]:

    """
    Extract cell images from the input image based on the specified bounding boxes.

    Parameters:
    -----------
    image (np.ndarray): Input image represented as a NumPy array.
    bboxes (List[np.ndarray]): List of bounding boxes (tuples) specifying the regions of interest.

    Returns:
    -------
    List[np.ndarray]: A list of cell images extracted from the input image based on the bounding boxes.

    """

    cells = []

    for b in bboxes:
         xmin,ymin,xmax,ymax = b  
         cell = image[int(ymin):int(ymax),int(xmin):int(xmax),:]
         cells.append(cell)

    return cells


def get_bounding_boxes_fastrcnn(mask: np.ndarray) -> List[Tuple]:
    """
    Get bounding boxes from a binary mask using region labeling.

    Parameters:
    -----------
    mask (np.ndarray): Binary mask representing the regions of interest.

    Returns:
    -------
    List[Tuple]: A list of bounding box tuples (xmin, ymin, xmax, ymax) extracted from the mask.
    """
  
    labeled = label(mask)
    props = regionprops(labeled)
    bboxes = []
    for prop in props:
        ymin,xmin,ymax,xmax = prop.bbox
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes

def get_processed_bboxes(bboxes: List[Tuple], size_image: Tuple) -> List[Tuple]:
    """
    Process bounding boxes to fit within the specified image size.

    Parameters:
    -----------
    bboxes (list of tuples): List of bounding box tuples (xmin, ymin, xmax, ymax).
    size_image (Tuple): Size of the target image in the format (width, height).

    Returns:
    -------
    List[Tuple]: A list of processed bounding box tuples that fit within the specified image size.

    """

    mask = np.zeros(size_image)#en formato numpy el size

    for bbox in bboxes:

        xmin,ymin,xmax,ymax = bbox

        top_left = (int(xmin),int(ymax))
        bottom_right = (int(xmax),int(ymin))

        cv2.rectangle(mask,top_left,bottom_right,255,-1)
    
    bboxes_processed = get_bounding_boxes_fastrcnn(mask)

    return bboxes_processed


def calculate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    -----------
    box_a (np.ndarray): Bounding box coordinates (x_min, y_min, x_max, y_max) of the first box.
    box_b (np.ndarray): Bounding box coordinates (x_min, y_min, x_max, y_max) of the second box.

    Returns:
    -------
    float: The Intersection over Union (IoU) score between the two bounding boxes.

    """

    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    
    intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

    
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = intersection_area / float(area_a + area_b - intersection_area)

    return iou

def find_duplicate_bounding_boxes(bounding_boxes1: List[np.ndarray], 
                                  candidates1: List[np.ndarray], 
                                  bounding_boxes2: List[np.ndarray], 
                                  candidates2: List[np.ndarray], 
                                  alpha: float = 0.5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Find duplicate bounding boxes between two sets of bounding boxes and their corresponding candidates.

    Parameters:
    -----------
    bounding_boxes1 (list): List of bounding box tuples (x_min, y_min, x_max, y_max) from the first set.
    candidates1 (list): List of candidate images corresponding to the bounding boxes in bounding_boxes1.
    bounding_boxes2 (list): List of bounding box tuples (x_min, y_min, x_max, y_max) from the second set.
    candidates2 (list): List of candidate images corresponding to the bounding boxes in bounding_boxes2.
    alpha (float, optional): Threshold value for Intersection over Union (IoU) to consider as duplicates.
                             Default is 0.5.

    Returns:
    -------
    tuple[list, list]: A tuple containing lists of duplicate images and their corresponding bounding boxes.

    """


    duplicate_boxes = []
    duplicate_images = []
    for i,box1 in enumerate(bounding_boxes1):
        for j,box2 in enumerate(bounding_boxes2):
            iou = calculate_iou(box1, box2)
            if iou > alpha:
                
                area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                if area_box1 >= area_box2:
                    duplicate_boxes.append(box1)
                    duplicate_images.append(candidates1[i])
                else:
                    duplicate_boxes.append(box2)
                    duplicate_images.append(candidates2[j])
    return duplicate_images,duplicate_boxes
