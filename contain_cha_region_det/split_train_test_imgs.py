'''
将原始数据划分为训练样本和测试样本
'''
import argparse
import os
import cv2
from tqdm import tqdm
import random
import math
import numpy as np
import shutil

def mkdir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdir_dir_(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            os.remove(file_path)

def main(opt):

    for name, num in opt.img_names_num_dict.items():
        print(name)
        img_dir = os.path.join(opt.img_dir, name + '/images')
        txt_dir = os.path.join(opt.img_dir, name + '/labels')

        save_train_dir = os.path.join(opt.save_dir, name + '/train')
        mkdir_dir(save_train_dir)
        save_test_dir = os.path.join(opt.save_dir, name + '/test')
        mkdir_dir(save_test_dir)

        save_train_img = os.path.join(save_train_dir, 'images')
        mkdir_dir_(save_train_img)

        save_train_labels = os.path.join(save_train_dir, 'labels')
        mkdir_dir_(save_train_labels)

        save_test_img = os.path.join(save_test_dir, 'images')
        mkdir_dir_(save_test_img)

        save_test_labels = os.path.join(save_test_dir, 'labels')
        mkdir_dir_(save_test_labels)


        imgs = os.listdir(img_dir)

        # 随机挑选
        list2 = [i for i in range(len(imgs))]
        choice_num_list = random.sample(list2, num)  # 100是需要制作的图片数量
        # dic_num_path = {str(i): path for (i, path) in zip(range(len(imgs)), imgs)}

        for img_i in tqdm(range(len(imgs))):
            img = imgs[img_i]
            img_path = os.path.join(img_dir, img)
            txt_path = os.path.join(txt_dir, img[:-4] + '.txt')
            # 随机挑选
            if img_i in choice_num_list:
                save_img_path = os.path.join(save_test_img, img)
                shutil.copy(img_path, save_img_path)
                save_txt_path = os.path.join(save_test_labels, img[:-4] + '.txt')
                shutil.copy(txt_path, save_txt_path)
            else:
                save_img_path = os.path.join(save_train_img, img)
                shutil.copy(img_path, save_img_path)
                save_txt_path = os.path.join(save_train_labels, img[:-4] + '.txt')
                shutil.copy(txt_path, save_txt_path)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    names_num_dict = {'935_936': 7}
    pathdir = '/home/wqg/data/container/'
    parser.add_argument('--img_dir', type=str,
                        default= pathdir + 'container_rec', help='')
    parser.add_argument('--img_names_num_dict', default=names_num_dict, help='')
    parser.add_argument('--save_dir', type=str,
                        default=pathdir + 'container_rec', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
