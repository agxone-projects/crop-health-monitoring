"""
Created on Wed Apr 24 2021

@author: chandralegend
"""

import argparse
import os
import cv2
import numpy as np
from imantics import Mask
from PIL import Image, ImageDraw
from pycocotools import mask as pymask
import json
from imantics import Mask
import datetime

from model import CornNetv2


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()


def img_resizer(imgdir, img_size=800):
    images = []
    image_file_names = [x for x in os.listdir(imgdir) if x.endswith('.jpg')]
    image_files = [os.path.join(imgdir, x) for x in image_file_names]
    for image in image_files:
        img = cv2.imread(image)
        img_resized = cv2.resize(
            img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        images.append(img_resized)
    return images, image_file_names


def output_to_mask(outputs, crop_size=800):
    instances_dict = outputs['instances'].get_fields()
    pred_masks = instances_dict['pred_masks'].tolist()
    pred_masks = [np.array(pred_mask) for pred_mask in pred_masks]
    anno_mask = sum(pred_masks) * 255
    if pred_masks == []:
        return np.array([[False] * crop_size] * crop_size) * 255
    return anno_mask


def whole_crop_check(crop_mask, crop_size=800, threshold=0.9):
    mask_area = cv2.countNonZero(crop_mask//255)
    if mask_area >= (crop_size**2) * threshold:
        white_crop = np.array([[True] * crop_size] * crop_size) * 255
        return white_crop
    return crop_mask


def get_segmentations(mask):
    annotations = Mask(mask).polygons().segmentation
    return annotations


def save_outputs(img, outputs, filename, savedir):
    mask_img = Image.new('RGB', (800, 800))
    mask = output_to_mask(outputs)
    mask = whole_crop_check(mask)/255
    mask = Image.fromarray(mask)
    mask_img.paste(mask, (0, 0))
    annotations = get_segmentations(mask)
    masked_output = cv2.bitwise_and(img, img, mask=cv2.cvtColor(
        np.array(mask_img), cv2.COLOR_BGR2GRAY))
    cv2.imwrite(os.path.join(savedir, f'mask_{filename}'), masked_output)
    return annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict the Segmentation Polygon of the Crops')
    parser.add_argument('--weights', required=True, type=str,
                        help="Relative path the .pth weights file [Required]")
    parser.add_argument('--thresh', default=0.7, type=float,
                        help="Score threshold for the prediction [0.7]")
    parser.add_argument('--imgdir', required=True, type=str,
                        help="Directory where the images are located")
    parser.add_argument('--savedir', default='./', type=str,
                        help="Save Directory for the Predictions")
    args = parser.parse_args()

    model = CornNetv2(args.weights, 1)
    predictor = model.get_predictor(args.thresh)

    images, filenames = img_resizer(args.imgdir)
    print(filenames)

    output_anno_dict = {"images": [],
                        "annotations": [],
                        "categories": [{
                            "id": 1,
                            "name": "crop",
                            "supercategory": "crop",
                            "color": "#0bbdcc"
                        }]}

    img_id = 1
    anno_id = 1

    for i, img in enumerate(images):
        outputs = predictor(img)
        annos = save_outputs(img, outputs, filenames[i], args.savedir)
        num_annotations = 0
        for anno in annos:
            seg_mask = Image.new('L', (800, 800), 0)
            ImageDraw.Draw(seg_mask).polygon(anno, fill=1)
            seg_mask = pymask.encode(np.asfortranarray(seg_mask))
            seg_area = pymask.area(seg_mask)
            seg_bbox = pymask.toBbox(seg_mask)
            crop_annotation = {
                "id": anno_id,
                "image_id": img_id,
                "category_id": 1,
                "width": 800,
                "height": 800,
                "segmentation": [anno],
                "iscrowd": 0,
                "isbbox": 0,
                "area": seg_area,
                "bbox": seg_bbox
            }
            output_anno_dict["annotations"].append(crop_annotation)
            anno_id += 1
            num_annotations += 1

        image = {
            "id": img_id,
            "category_ids": [1],
            "width": 800,
            "height": 800,
            "file_name": filenames[i],
            "num_annotations": num_annotations
        }
        output_anno_dict["images"].append(image)
        img_id += 1

    with open(os.path.join(args.savedir, 'output_annotations.json'), 'wt', encoding='UTF-8') as anno_file:
        json.dump(output_anno_dict, anno_file, indent=2,
                  sort_keys=True, default=myconverter)
