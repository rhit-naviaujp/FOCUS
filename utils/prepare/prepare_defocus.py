#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import json
import os
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
import tqdm
from pathlib import Path

def create_instance_dataset(dataset_dir, output_dir, dataset_type):
    """
    Convert a COD dataset to COCO format for specified dataset type (Test or Train).
    Args:
    - dataset_dir: Directory containing the COD dataset.
    - output_dir: Directory to save the COCO formatted dataset.
    - dataset_type: 'Test' or 'Train'.
    """
    if dataset_type == 'TRAIN':
        image_dir = os.path.join(dataset_dir, f"train_data/1204source")
        mask_dir = os.path.join(dataset_dir, f"train_data/1204gt")
    elif dataset_type == 'DUT':
        image_dir = os.path.join(dataset_dir, f"test_data/DUT/dut500-source")
        mask_dir = os.path.join(dataset_dir, f"test_data/DUT/dut500-gt")
    elif dataset_type == 'CUHK':
        image_dir = os.path.join(dataset_dir, f"test_data/CUHK/xu100-source")
        mask_dir = os.path.join(dataset_dir, f"test_data/CUHK/xu100-gt")
    pre_process(mask_dir)

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.bmp")))
    mask_files = sorted([os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + ".bmp") for f in image_files])

    images = []
    annotations = []

    ann_id = 1

    for img_id, (image_file, mask_file) in tqdm.tqdm(enumerate(zip(image_files, mask_files), start=1), total=len(image_files)):
        # Process image information
        file_name = os.path.basename(image_file)
        image = Image.open(image_file)
        width, height = image.size

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # Process mask information
        mask = np.array(Image.open(mask_file).convert("L"))
        binary_mask = np.where(mask > 128, 255, 0).astype(np.uint8)
        # Convert binary mask to RLE
        rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        
        rle['counts'] = rle['counts'].decode('utf-8')

        bbox = mask_util.toBbox(rle).tolist()
        area = mask_util.area(rle).tolist()

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "file_name": file_name,
            "category_id": 1,  # Assuming a single category for salient objects
            "segmentation": rle,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        })
        ann_id += 1

    # Categories (assuming a single generic category for salient objects)
    categories = [{
        "id": 1,
        "name": "foreground",
        "supercategory": "foreground"
    }]

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save to file
    output_file_path = os.path.join(output_dir, f"DEFOCUS_{dataset_type}_INSTANCE.json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file_path, 'w') as f:
        json.dump(coco_format, f)
    print(f"DEFOCUS-{dataset_type} in COCO format is saved to {output_file_path}")
def create_semantic_dataset(dataset_dir, output_dir, dataset_type):
    """
    Convert a COD dataset to COCO format for specified dataset type (Test or Train).
    Args:
    - dataset_dir: Directory containing the COD dataset.
    - output_dir: Directory to save the COCO formatted dataset.
    - dataset_type: 'Test' or 'Train'.
    """
    if dataset_type == 'TRAIN':
        image_dir = os.path.join(dataset_dir, f"train_data/1204source")
        mask_dir = os.path.join(dataset_dir, f"train_data/1204gt")
    elif dataset_type == 'DUT':
        image_dir = os.path.join(dataset_dir, f"test_data/DUT/dut500-source")
        mask_dir = os.path.join(dataset_dir, f"test_data/DUT/dut500-gt")
    elif dataset_type == 'CUHK':
        image_dir = os.path.join(dataset_dir, f"test_data/CUHK/xu100-source")
        mask_dir = os.path.join(dataset_dir, f"test_data/CUHK/xu100-gt")

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.bmp")))
    mask_files = sorted([os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + ".bmp") for f in image_files])

    images = []
    annotations = []

    ann_id = 1

    for img_id, (image_file, mask_file) in tqdm.tqdm(enumerate(zip(image_files, mask_files), start=1), total=len(image_files)):
        # Process image information
        file_name = os.path.basename(image_file)
        image = Image.open(image_file)
        width, height = image.size

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # Process mask information
        mask = np.array(Image.open(mask_file).convert("L"))
        binary_mask = mask != 0  # Non-zero pixels mark the object

        background_mask = mask == 0

        # Convert binary mask to RLE
        rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')

        background_rle = mask_util.encode(np.asfortranarray(background_mask.astype(np.uint8)))
        background_rle['counts'] = background_rle['counts'].decode('utf-8')

        bg_bbox = mask_util.toBbox(background_rle).tolist()
        bg_area = mask_util.area(background_rle).tolist()



        bbox = mask_util.toBbox(rle).tolist()
        area = mask_util.area(rle).tolist()

        annotations.append({
            "image_id": img_id,
            "file_name": file_name,
            "segments_info": [{
                "id": ann_id,
                "category_id": 1,  # '2' for object; '1' would typically be for background, but is not used here
                # "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            },{
                "id": ann_id+1,
                "category_id": 2,  
                # "segmentation": background_rle,
                "area": bg_area,
                "bbox": bg_bbox,
                "iscrowd": 0
            }]

        })
        ann_id += 2

    # Categories
    categories = [
        {"id": 1, "name": "foreground", "supercategory": "foreground","is_thing": 1},
        {"id": 2, "name": "background", "supercategory": "background","is_thing": 0}
    ]

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save to file
    output_file_path = os.path.join(output_dir, f"DEFOCUS_{dataset_type}_SEMANTIC.json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file_path, 'w') as f:
        json.dump(coco_format, f)
    print(f"DEFOCUS-{dataset_type} in COCO format is saved to {output_file_path}")
def pre_process(directory):
    files = [filename for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename)) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for filename in tqdm.tqdm(files, desc="Processing images", unit="image"):
        file_path = os.path.join(directory, filename)
        
        img = Image.open(file_path)

        
        pixels = list(img.getdata())
        
        img = img.point(lambda p: 255 if p > 128 else 0)
        if img.mode != 'L':
            img = img.convert("L").point(lambda p: 255 if p > 128 else 0)
        img.save(file_path)
def convert(input, output):
    img = np.asarray(Image.open(input)).astype(np.uint8)
    assert img.dtype == np.uint8
    img = img + 1  # . others are shifted by 1
    Image.fromarray(img).save(output)


if __name__ == "__main__":
    dataset_dir = "datasets/DEFOCUS/dataset"  # Update this to the path of your dataset
    output_dir = "datasets/DEFOCUS/dataset"  # Update this to your desired output directory
    for dataset_type in ['TRAIN', 'DUT','CUHK']:
        create_instance_dataset(dataset_dir, output_dir, dataset_type)
        create_semantic_dataset(dataset_dir, output_dir, dataset_type)


    for name in ["test_data/CUHK/xu100-gt", "test_data/DUT/dut500-gt","train_data/1204gt"]:
        annotation_dir = Path(os.path.join(dataset_dir ,  name))
        output_dir = Path(os.path.join(dataset_dir , "annotations_detectron2" , name.split('/')[-1]))
        print(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            if str(file).endswith(('.png', '.jpg', '.jpeg',"bmp")):
                convert(file, output_file)