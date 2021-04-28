"""
Created on Wed Apr 21 13:54:07 2021

@author: chandralegend
"""

import wget
import cv2
import numpy as np
import datetime

from pycocotools.coco import COCO
from PIL import Image, ImageDraw
from pycocotools import mask as pymask
import json
from imantics import Mask


def download_dataset():
    print("Downloading Stitched Image from\n 'https://archive.org/download/stitched_image_202104/stitched_image.jpg'")
    wget.download('https://archive.org/download/stitched_image_202104/stitched_image.jpg',
                  './data/raw/stitched_image.jpg')
    print("Downloading Annotation from\n 'https://archive.org/download/cornnetv2_annotations/cornnetv2_annotations.json'")
    wget.download('https://archive.org/download/cornnetv2_annotations/cornnetv2_annotations.json',
                  './data/raw/annotations.json')


def get_full_mask_array(annos, width, height):
    mask = Image.new('L', (width, height), 0)
    for anno in annos:
        segs = anno['segmentation']
        for seg in segs:
            ImageDraw.Draw(mask).polygon([x * 4 for x in seg], fill=1)
    return np.array(mask)


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()


def create_crops(stitch_img, stitch_anno, crop_size, dest_dir):

    stitch_anno_obj = COCO(stitch_anno)
    stitch_annos = stitch_anno_obj.loadAnns(stitch_anno_obj.getAnnIds())

    img_height, img_width = stitch_img.shape[:2]
    full_mask_array = get_full_mask_array(stitch_annos, width, height)

    new_anno_dict = {"images": [],
                     "annotations": [],
                     "categories": [{
                         "id": 1,
                         "name": "crop",
                         "supercategory": "crop",
                         "color": "#0bbdcc"
                     }]}

    crop_id = 1
    anno_id = 1

    for i in range(img_height // crop_size):
        for j in range(img_width // crop_size):
            x1, y1 = (crop_size * j), (crop_size * i)
            x2, y2 = (x1 + crop_size), (y1 + crop_size)

            cv2.imwrite(f'{dest_dir}/crop_{i}_{j}.jpg',
                        stitch_img[y1:y2, x1:x2])

            crop_mask_array = np.array(full_mask_array[y1:y2, x1:x2])

            annotations = Mask(crop_mask_array).polygons().segmentation
            num_annotations = 0
            for anno in annotations:
                if (len(anno) != 2):
                    seg_mask = Image.new('L', (crop_size, crop_size), 0)
                    ImageDraw.Draw(seg_mask).polygon(anno, fill=1)
                    seg_mask = pymask.encode(np.asfortranarray(seg_mask))
                    seg_area = pymask.area(seg_mask)
                    seg_bbox = pymask.toBbox(seg_mask)
                    crop_annotation = {
                        "id": anno_id,
                        "image_id": crop_id,
                        "category_id": 1,
                        "width": crop_size,
                        "height": crop_size,
                        "segmentation": [anno],
                        "iscrowd": 0,
                        "isbbox": 0,
                        "area": seg_area,
                        "bbox": seg_bbox
                    }
                    new_anno_dict["annotations"].append(crop_annotation)
                    anno_id += 1
                    num_annotations += 1

            crop_image = {
                "id": crop_id,
                "category_ids": [1],
                "width": crop_size,
                "height": crop_size,
                "file_name": f'crop_{i}_{j}.jpg',
                "num_annotations": num_annotations
            }
            new_anno_dict["images"].append(crop_image)
            crop_id += 1

    with open(f'{dest_dir}/annotations.json', 'wt', encoding='UTF-8') as anno_file:
        json.dump(new_anno_dict, anno_file, indent=2,
                  sort_keys=True, default=myconverter)


if __name__ == '__main__':
    download_dataset()

    Image.MAX_IMAGE_PIXELS = None

    stitch_img = np.array(Image.open('./data/raw/stitched_image.jpg'))
    stitch_anno = './data/raw/annotations.json'
    destination = './data/interim'

    stitch_anno_obj = COCO(stitch_anno)
    annos = stitch_anno_obj.loadAnns(stitch_anno_obj.getAnnIds())
    height, width = stitch_img.shape[:2]
    crop_size = 800

    create_crops(stitch_img, stitch_anno, crop_size, destination)
