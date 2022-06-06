import os
import shutil

import cv2
import tqdm
import numpy as np
import pdb
import json, argparse


def calc_k(line):
    '''
    Calculate the direction of lanes
    '''
    line_x = line[::2]
    line_y = line[1::2]
    length = np.sqrt((line_x[0] - line_x[-1]) ** 2 + (line_y[0] - line_y[-1]) ** 2)
    if length < 90:
        return -10  # if the lane is too short, it will be skipped

    p = np.polyfit(line_x, line_y, deg=1)
    rad = np.arctan(p[0])

    return rad


def draw(im, line, idx, shape, show=False):
    '''
    Generate the segmentation label according to json annotation
    '''
    line_x = line[::2]
    line_y = line[1::2]
    pt0 = (int((line_x[0])/shape[1] *640), int(line_y[0]/shape[0] *360))
    if show:
        cv2.putText(im, str(idx), (int(line_x[len(line_x) // 2]), int(line_y[len(line_x) // 2]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = idx * 60

    for i in range(len(line_x) - 1):
        cv2.line(im, pt0, (int(line_x[i + 1]/shape[1] *640), int(line_y[i + 1]/shape[0] *360)), (idx,), thickness=2)  # thickness = i//4 +1
        pt0 = (int(line_x[i + 1]/shape[1] *640), int(line_y[i + 1]/shape[0] *360))
    # cv2.imshow('im', im)
    # cv2.waitKey(0)


def get_CULane_list(root):
    '''
    Get all the files' names from the json annotation
    '''
    label_all = []
    img_all = []

    fileList = os.listdir(os.path.join(root, 'moive'))
    for files in fileList:
        for file in os.listdir(os.path.join(root, 'moive',files,'image')):
            if os.path.splitext(file)[-1] == '.jpg':
                labelname = file.replace('.jpg', '.lines.json')
                label_all.append(os.path.join(root, 'moive', files, 'label', labelname))
                img_all.append(os.path.join(root, 'moive', files, 'image', file))

    return img_all, label_all


def get_CurveLanes_point(fliename):
    with open(fliename, 'r') as file:
        info_dict = json.load(file)
        file = info_dict['Lines']
        if file == []:
            return []

        line_txt = []
        for lane_index, lane in enumerate(file):
            # line_txt_i = []
            # lane = lane.split()
            lane_x = [float(xinfo['x']) for xinfo in lane]
            lane_y = [float(yinof['y']) for yinof in lane]
            line_txt_tmp = [None] * (len(lane_y) + len(lane_x))
            line_txt_tmp[::2] = list(map(str, lane_x))
            line_txt_tmp[1::2] = list(map(str, lane_y))
            line_txt.append(line_txt_tmp)
        # line_txt.append(line_txt_i)

        # lane_pts = np.vstack((lane_x, lane_y)).transpose()
        # lane_pts = np.array([lane_pts], np.int64)

    return line_txt


def generate_segmentation_and_train_list(root, imgList, labelList):  # , line_txt, names
    """
    The lane annotations of the Tusimple dataset is not strictly in order, so we need to find out the correct lane order for segmentation.
    We use the same definition as CULane, in which the four lanes from left to right are represented as 1,2,3,4 in segentation label respectively.
    """


    train_gt_fp = open(os.path.join(root,  'train_gt.txt'), 'w')

    # fileList = os.listdir(os.path.join(root,  'images'))
    for idx,imgname in enumerate(imgList[:200]):
        tmp_line = get_CurveLanes_point(labelList[idx])

        lines = []
        for j in range(len(tmp_line)):
            lines.append(list(map(float, tmp_line[j])))

        ks = np.array([calc_k(line) for line in lines])  # get the direction of each lane

        k_neg = ks[ks < 0].copy()
        k_pos = ks[ks > 0].copy()
        k_neg = k_neg[k_neg != -10]  # -10 means the lane is too short and is discarded
        k_pos = k_pos[k_pos != -10]
        k_neg.sort()
        k_pos.sort()
        k_neg = list(k_neg)
        k_pos = list(k_pos)
        kn_l = len(k_neg)
        kp_l = len(k_pos)

        ke = 0.15

        # 2条线隔的太近删除靠外侧一条.
        if kn_l >= 2:
            for i in range(kn_l - 1):
                p1 = kn_l - 1 - i
                p2 = kn_l - 2 - i
                if abs(k_neg[p1] - k_neg[p2]) < ke:
                    k_neg.pop(p1)
        if kp_l >= 2:
            for i in range(kp_l - 1):
                p1 = kp_l - 1 - i
                p2 = kp_l - 2 - i
                if abs(k_pos[p1] - k_pos[p2]) < ke:
                    k_pos.pop(p2)




        name = imgname.split('/')[-1][:-4]
        img = cv2.imread(imgname)
        shape = img.shape  # shape -> h,w,c
        # if shape[0] != 590 or shape[1] != 1640:
        #     print(imgname)

        label = np.zeros((360, 640), dtype=np.uint8)
        img = cv2.resize(img, dsize=(640, 360))
        # img = label

        bin_label = [0, 0, 0, 0]
        if len(k_neg) == 1:  # for only one lane in the left
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label, lines[which_lane], 2,shape)
            bin_label[1] = 1
        elif len(k_neg) >= 2:  # for two lanes in the left

            which_lane = np.where(ks == k_neg[1])[0][0]
            draw(label, lines[which_lane], 1,shape)
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label, lines[which_lane], 2,shape)
            bin_label[0] = 1
            bin_label[1] = 1

        if len(k_pos) == 1:  # For the lanes in the right, the same logical is adopted.
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label, lines[which_lane], 3,shape)
            bin_label[2] = 1
        elif len(k_pos) >= 2:
            which_lane = np.where(ks == k_pos[len(k_pos) - 1])[0][0]
            draw(label, lines[which_lane], 3,shape)
            which_lane = np.where(ks == k_pos[len(k_pos) - 2])[0][0]
            draw(label, lines[which_lane], 4,shape)
            bin_label[2] = 1
            bin_label[3] = 1

        labelpath = os.path.join(root,  'auto_gt_image', name+'.png')

        cv2.imwrite(labelpath, label)


        cv2.imwrite(os.path.join(root,  'auto_images', name+'.jpg'), img)

        train_gt_fp.write(os.path.join('CurveLanes',  'auto_image', name+'.jpg') + ' ' + os.path.join('CurveLanes'+ 'auto_gt_image', name+'.png') + ' ' + ' '.join(list(map(str, bin_label))) + '\n')
    train_gt_fp.close()


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--root',default='/home/wqg/data/LANE DETECTION CHALLENGE/train_set', required=True, help='The root of the Tusimple dataset')
    parser.add_argument('--root', default='/home/wqg/data/CurveLanes', help='The root of the Tusimple dataset')
    return parser


if __name__ == "__main__":
    args = get_args().parse_args()

    # generate segmentation and training list for training
    os.makedirs(os.path.join(args.root, 'auto_images'), exist_ok=True)
    os.makedirs(os.path.join(args.root, 'auto_gt_image'), exist_ok=True)
    imgList, labelList = get_CULane_list(args.root)

    generate_segmentation_and_train_list(args.root, imgList, labelList)