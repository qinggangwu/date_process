#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
整理 CULane 车道线公开数据集
by : wuqinggang

image_size : 1640*590

1. Adding valiation set/test set
2. Fix some small error

"""

import argparse
import glob
import json
import os
import os.path as ops
import shutil
from tqdm import tqdm
import cv2
import numpy as np

# file_index = 58785
file_index = 0

def init_args():
    """
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='The origin path of unzipped tusimple dataset',default= '/home/wqg/data/CULane')
    parser.add_argument('--val', type=bool, help='Tag for validation set', default=True)
    parser.add_argument('--test', type=bool, help='Tag for validation set', default=False)

    return parser.parse_args()

def process_txt_file(txt_file_path, src_dir, ori_dst_dir, binary_dst_dir, instance_dst_dir):
    """
    :param txt_file_path:
    :param src_dir: origin clip file path
    :param ori_dst_dir:
    :param binary_dst_dir:
    :param instance_dst_dir:
    :return:
    """
    assert ops.exists(txt_file_path), '{:s} not exist'.format(txt_file_path)
    # image_nums = len(os.listdir(ori_dst_dir))

    with open(txt_file_path, 'r') as file:
        f = False

        image_path = txt_file_path.replace('.lines.txt', '.jpg')
        if not ops.exists(image_path):
            print('{:s} not exist'.format(image_path))
            return

        src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        dst_binary_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)
        dst_instance_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)

        for lane_index, lane in enumerate(file):
            lane = lane.split()
            lane_x = [ int(float(lane[i])) for i in range(len(lane)) if i %2 == 0 ]
            lane_y = [ int(lane[i]) for i in range(len(lane)) if i %2 == 1 ]

            lane_pts = np.vstack((lane_x, lane_y)).transpose()
            lane_pts = np.array([lane_pts], np.int64)

            cv2.polylines(dst_binary_image, lane_pts, isClosed=False,  color=255, thickness=5)
            cv2.polylines(dst_instance_image, lane_pts, isClosed=False, color=lane_index * 50 + 20, thickness=5)
            f = True

        if f and ops.exists(image_path):
            global file_index
            file_index +=1
            image_name_new = '{:s}.png'.format('{:d}'.format(file_index).zfill(6))
            dst_binary_image_path = ops.join(binary_dst_dir, image_name_new)
            dst_instance_image_path = ops.join(instance_dst_dir, image_name_new)
            dst_rgb_image_path = ops.join(ori_dst_dir, image_name_new[:-3]+'jpg')

            cv2.imwrite(dst_binary_image_path, dst_binary_image)
            cv2.imwrite(dst_instance_image_path, dst_instance_image)
            cv2.imwrite(dst_rgb_image_path, src_image)

            # print('Process {:s} success'.format(image_name_new))

def gen_train_sample(src_dir, b_gt_image_dir, i_gt_image_dir, image_dir):
    """
    generate sample index file
    :param src_dir:
    :param b_gt_image_dir:
    :param i_gt_image_dir:
    :param image_dir:
    :return:
    """

    with open('{:s}/training/train.txt'.format(src_dir), 'w') as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.png'):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name.replace('png','jpg'))

            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(instance_gt_image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None or i_gt_image is None:
                print('图像对: {:s}损坏'.format(image_name))
                continue
            else:
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path)
                file.write(info + '\n')
    return

def gen_train_val_sample(src_dir, b_gt_image_dir, i_gt_image_dir, image_dir):
    """
    generate sample index file
    val : train= 1:12   拆分比例

    :param src_dir:
    :param b_gt_image_dir:
    :param i_gt_image_dir:
    :param image_dir:
    :return:
    """
    with open('{:s}/training/train.txt'.format(src_dir), 'w') as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.png'):
                continue
            if(int(image_name.split('.')[0]) % 13 == 0):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name.replace('png','jpg'))

            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(instance_gt_image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None or i_gt_image is None:
                print('图像对: {:s}损坏'.format(image_name))
                continue
            else:
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path)
                file.write(info + '\n')

    with open('{:s}/training/val.txt'.format(src_dir), 'w') as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.png'):
                continue
            if(int(image_name.split('.')[0]) % 13 != 0):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(instance_gt_image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None or i_gt_image is None:
                print('图像对: {:s}损坏'.format(image_name))
                continue
            else:
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path)
                file.write(info + '\n')
    return

def gen_test_sample(src_dir, b_gt_image_dir, i_gt_image_dir, image_dir):
    """
    generate sample index file
    :param src_dir:
    :param b_gt_image_dir:
    :param i_gt_image_dir:
    :param image_dir:
    :return:
    """

    with open('{:s}/testing/test.txt'.format(src_dir), 'w') as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.png'):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(instance_gt_image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None or i_gt_image is None:
                print('图像对: {:s}损坏'.format(image_name))
                continue
            else:
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path)
                file.write(info + '\n')
    return

def process_tusimple_dataset(src_dir, val_tag, test_tag):
    """
    :param src_dir:
    :return:
    """
    training_folder_path = ops.join(src_dir, 'train')
    testing_folder_path = ops.join(src_dir, 'testing')

    os.makedirs(training_folder_path, exist_ok=True)
    os.makedirs(testing_folder_path, exist_ok=True)

    gt_image_dir = ops.join(training_folder_path, 'gt_image')
    gt_binary_dir = ops.join(training_folder_path, 'gt_binary_image')
    gt_instance_dir = ops.join(training_folder_path, 'gt_instance_image')

    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)

    fileList = ['driver_23_30frame','driver_37_30frame','driver_100_30frame','driver_161_90frame','driver_182_30frame','driver_193_90frame']
    # fileList = ['driver_100_30frame','driver_161_90frame','driver_182_30frame','driver_193_90frame']

    for file in fileList:
        for file_ in tqdm(os.listdir(ops.join(src_dir,file))):
            # if file_ in ['05252023_0511.MP4','05252249_0542.MP4','05251251_0391.MP4'] or\
            #         file_ in ['05250322_0271.MP4','05251149_0374.MP4', '05252120_0524.MP4','05250948_0350.MP4','05252116_0523.MP4',
            #                   '05250340_0277.MP4',]:
            #     continue
            print(ops.join(src_dir,file,file_))
            for ind,txt_label_path in enumerate(glob.glob('{:s}/*.txt'.format(ops.join(src_dir,file,file_)))):
                process_txt_file(txt_label_path, src_dir, gt_image_dir, gt_binary_dir, gt_instance_dir)

    if (val_tag == False):
        gen_train_sample(src_dir, gt_binary_dir, gt_instance_dir, gt_image_dir)
    else:
        gen_train_val_sample(src_dir, gt_binary_dir, gt_instance_dir, gt_image_dir)

    if (test_tag == True):
        gt_image_dir_test = ops.join(testing_folder_path, 'gt_image')
        gt_binary_dir_test = ops.join(testing_folder_path, 'gt_binary_image')
        gt_instance_dir_test = ops.join(testing_folder_path, 'gt_instance_image')

        os.makedirs(gt_image_dir_test, exist_ok=True)
        os.makedirs(gt_binary_dir_test, exist_ok=True)
        os.makedirs(gt_instance_dir_test, exist_ok=True)

        for json_label_path in glob.glob('{:s}/*.json'.format(testing_folder_path)):
            process_txt_file(json_label_path, src_dir, gt_image_dir_test, gt_binary_dir_test, gt_instance_dir_test)

        gen_test_sample(src_dir, gt_binary_dir_test, gt_instance_dir_test, gt_image_dir_test)

    return

if __name__ == '__main__':
    args = init_args()

    process_tusimple_dataset(args.src_dir, args.val, args.test)

