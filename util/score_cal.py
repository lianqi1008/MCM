import cv2
import numpy as np
import os
import csv
import torchvision.transforms.functional as f
import argparse

parser = argparse.ArgumentParser(description='script to compute scores')
parser.add_argument('-d0','--dir_origin', default='./dataset/coco/images/test', help='Path to original images', type=str)
parser.add_argument('-d1','--dir_structure', default='./dataset/coco/structure/test', help='Path to structure images', type=str)
parser.add_argument('-d2','--dir_score', default='./dataset/coco/scores/test', help='Path to output score', type=str)

opt = parser.parse_args()

def laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return mask_img

def cal_img(img, crop_sz=16, step=16):
    h, w, c = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    patch_scores = []
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            score = laplacian(crop_img).mean()
            score = round(score, 2)
            patch_scores.append(score)
    return patch_scores

def cal_structure(img, crop_sz=16, step=16):
    h, w, c = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    patch_scores = []
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[x:x + crop_sz, y:y + crop_sz,:]
            score = int(crop_img.sum()//255)
            patch_scores.append(score)
    return patch_scores

if __name__ == '__main__':
    img_names = sorted(os.listdir(opt.dir_origin))
    label_names = sorted(os.listdir(opt.dir_structure))
    with open(os.path.join(opt.dir_score, 'texture.csv'), 'a', encoding='utf-8', newline='') as f:
        with open(os.path.join(opt.dir_score, 'structure.csv'), 'a', encoding='utf-8', newline='') as f1:
            cw = csv.writer(f)
            cw1 = csv.writer(f1)

            patch_list = list(range(256))
            patch_list.insert(0, 'image')

            structure_list = list(range(256))
            structure_list.insert(0, 'image')

            cw.writerows([patch_list])
            cw1.writerows([structure_list])

            for i in range(len(img_names)):
                if i % 50 == 0:
                    print(i + 1)
                img_name = img_names[i]
                label_name = label_names[i]

                img_path = os.path.join(opt.dir_origin, img_name)
                label_path = os.path.join(opt.dir_structure, label_name)

                image = cv2.imread(img_path)
                label = cv2.imread(label_path)

                patch_scores = cal_img(image, 16, 16)
                structure_scores = cal_structure(label, 16, 16)

                patch_scores.insert(0, img_name)
                cw.writerows([patch_scores])

                structure_scores.insert(0, img_name)
                cw1.writerows([structure_scores])

    f.close()
    f1.close()