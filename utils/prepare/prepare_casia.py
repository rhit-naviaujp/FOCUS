#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import json
import os
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm
from pathlib import Path
import shutil
def create_instance_dataset(dataset_dir, output_dir, dataset_type):
    """
    Convert a COD dataset to COCO format for specified dataset type (TEST or TRAIN).
    Args:
    - dataset_dir: Directory containing the COD dataset.
    - output_dir: Directory to save the COCO formatted dataset.
    - dataset_type: 'TEST' or 'TRAIN'.
    """
    image_dir = os.path.join(dataset_dir, f"{dataset_type}/image")
    mask_dir = os.path.join(dataset_dir, f"{dataset_type}/gt")
    pre_process(mask_dir)

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))


    mask_files = sorted([os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + "_gt.png") for f in image_files])
    images = []
    annotations = []

    ann_id = 1

    for img_id, (image_file, mask_file) in enumerate(tqdm(zip(image_files, mask_files), total=len(image_files), desc="Processing images and masks", unit="pair"), start=1):
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
        # print(mask_file)
        mask = np.array(Image.open(mask_file).convert("L"))
        mheight,mwidth = mask.shape
        if (width,height) != (mwidth,mheight):
            
            # print(f"Image and mask size mismatch for {file_name},({width},{height}),({mwidth},{mheight}).")
            continue
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
    output_file_path = os.path.join(output_dir, f"CASIA_{dataset_type}.json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file_path, 'w') as f:
        json.dump(coco_format, f)
    print(f"COCO dataset for {dataset_type} saved to {output_file_path}")

def create_semantic_dataset(dataset_dir, output_dir, dataset_type):
    """
    Convert a COD dataset to COCO format for specified dataset type (TEST or TRAIN).
    Args:
    - dataset_dir: Directory containing the COD dataset.
    - output_dir: Directory to save the COCO formatted dataset.
    - dataset_type: 'TEST' or 'TRAIN'.
    """

    image_dir = os.path.join(dataset_dir, f"{dataset_type}/image")
    mask_dir = os.path.join(dataset_dir, f"{dataset_type}/gt")

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))


    mask_files = sorted([os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + "_gt.png") for f in image_files])

    images = []
    annotations = []

    ann_id = 1

    for img_id, (image_file, mask_file) in enumerate(tqdm(zip(image_files, mask_files), total=len(image_files), desc="Processing images and masks", unit="pair"), start=1):
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
        mheight,mwidth = mask.shape
        if (width,height) != (mwidth,mheight):
            
            # print(f"Image and mask size mismatch for {file_name},({width},{height}),({mwidth},{mheight}).")
            continue

        # print(mask_file)
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
    output_file_path = os.path.join(output_dir, f"CASIA_SEMSNTIC_{dataset_type}.json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file_path, 'w') as f:
        json.dump(coco_format, f)
    print(f"COCO dataset for {dataset_type} saved to {output_file_path}")
def merge_folders(src_folder1, src_folder2, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    src_files1 = [os.path.join(src_folder1, f) for f in os.listdir(src_folder1)]
    for file in src_files1:
        if os.path.isfile(file):
            shutil.copy(file, dest_folder)

    src_files2 = [os.path.join(src_folder2, f) for f in os.listdir(src_folder2)]
    for file in src_files2:
        if os.path.isfile(file):
            shutil.copy(file, dest_folder)

def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img + 1  # . others are shifted by 1
    Image.fromarray(img).save(output)
def pre_process(directory):
    files = [filename for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename)) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for filename in tqdm(files, desc="Processing images", unit="image"):
        file_path = os.path.join(directory, filename)
        
        img = Image.open(file_path)

        
        pixels = list(img.getdata())
        
        img = img.point(lambda p: 255 if p > 128 else 0)
        if img.mode != 'L':
            img = img.convert("L").point(lambda p: 255 if p > 128 else 0)
        img.save(file_path)

if __name__ == "__main__":



    # image_folder = "datasets/CASIA/CASIA 1.0 dataset/Tp/"

    # cm_folder = os.path.join(image_folder, "CM")
    # sp_folder = os.path.join(image_folder, "Sp")
    # output_folder = "datasets/CASIA/TEST/image"

    # merge_folders(cm_folder, sp_folder, output_folder)

    # gt_folder = "datasets/CASIA/CASIA 1.0 groundtruth"
    # cm_folder_gt = os.path.join(gt_folder, "CM")
    # sp_folder_gt = os.path.join(gt_folder, "Sp")
    # output_folder_gt = "datasets/CASIA/TEST/gt"

    # merge_folders(cm_folder_gt, sp_folder_gt, output_folder_gt)


    # # au_folder = "datasets/CASIA/CASIA2.0_revised/Au"
    # # tp_folder = "datasets/CASIA/CASIA2.0_revised/Tp"
    # # output_train_folder_image = "datasets/CASIA/TRAIN/image"

    # # merge_folders(au_folder, tp_folder, output_train_folder_image)

    # src_folder = "datasets/CASIA/CASIA2.0_revised/Tp"
    # dest_folder = "datasets/CASIA/TRAIN/image"

    # if not os.path.exists(dest_folder):
    #     os.makedirs(dest_folder)

    # for filename in os.listdir(src_folder):
    #     if filename.lower().endswith(('.tif', '.jpg', '.jpeg', '.bmp', '.png', '.gif')):
    #         src_file = os.path.join(src_folder, filename)
    #         dest_file = os.path.join(dest_folder, f"{os.path.splitext(filename)[0]}.jpg")
            
    #         with Image.open(src_file) as img:
    #             if img.mode == 'RGBX':
    #                 img = img.convert('RGBA')
    #             img.save(dest_file, "PNG")

    # src_folder_gt = "datasets/CASIA/CASIA2.0_Groundtruth"
    # dest_folder_gt = "datasets/CASIA/TRAIN/gt"

    # if not os.path.exists(dest_folder_gt):
    #     os.makedirs(dest_folder_gt)

    # src_files = [os.path.join(src_folder_gt, f) for f in os.listdir(src_folder_gt)]
    # for file in src_files:
    #     if os.path.isfile(file):
    #         shutil.copy(file, dest_folder_gt) 
    # file_path = "datasets/CASIA/TEST/image/Sp_D_NRN_A_cha0011_sec0011_0542.jpg"
    # if os.path.exists(file_path):
    #     os.remove(file_path)


    root = os.getenv("DETECTRON2_DATASETS", "datasets")
    dataset_dir = os.path.join(root,"CASIA")  # Update this to the path of your dataset
    output_dir = os.path.join(root,"CASIA")  # Update this to your desired output directory
    for dataset_type in ['TRAIN',"TEST"]:
        create_instance_dataset(dataset_dir, output_dir, dataset_type)
        create_semantic_dataset(dataset_dir, output_dir, dataset_type)
    

    for name in ["TEST", "TRAIN"]:
        annotation_dir = Path(os.path.join(dataset_dir ,  name, "gt"))
        output_dir = Path(os.path.join(dataset_dir , "annotations_detectron2" , name))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for file in tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file)