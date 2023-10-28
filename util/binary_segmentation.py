"""
get semantic segmentation annotations from coco data set.
"""
import argparse
import os
import tqdm
from pycocotools.coco import COCO
import cv2
import random
import numpy as np

def main(args):
    os.makedirs(os.path.join(args.image_dir), exist_ok=True)
    os.makedirs(os.path.join(args.segm_dir), exist_ok=True)

    coco = COCO(args.annotation_file)
    catIds = coco.getCatIds() 
    imgIds = coco.getImgIds()
    
    # Random sample 200 images for test
    if "val" in args.coco_dir:
        imgIds = [imgId for imgId in imgIds if len(coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)) > 0]
        imgIds = random.sample(imgIds, 200)
        print("Generate images and annotations for testing!")
    else:
        print("Generate images and annotations for training!")
    

    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            mask[mask > 0] = 255

            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
                mask[(mask + coco.annToMask(anns[i + 1])) > 0] = 255
            img_origin_path = os.path.join(args.coco_dir, img['file_name'])
            img_output_path = os.path.join(args.image_dir, img['file_name'])
            seg_output_path = os.path.join(args.segm_dir, img['file_name'])

            # save image
            image = cv2.imread(img_origin_path)
            image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(img_output_path, image)

            # save binary mask
            mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(seg_output_path, mask)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", default="/opt/data/private/laq/dataset/coco/val2014", type=str, help="COCO train/test dataset directory")
    parser.add_argument("--annotation_file", default="/opt/data/private/laq/dataset/coco/annotations/instances_val2014.json", type=str, help="COCO annotation file path")

    parser.add_argument("--image_dir", default="./dataset/coco/images/test", type=str, help="resized COCO train/test dataset directory")
    parser.add_argument("--segm_dir", default="./dataset/coco/structure/test", type=str, help="binary COCO train/test segmentation mask directory")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)