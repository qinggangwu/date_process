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
    length = np.sqrt((line_x[0]-line_x[-1])**2 + (line_y[0]-line_y[-1])**2)
    if length < 90:
        return -10                                          # if the lane is too short, it will be skipped

    p = np.polyfit(line_x, line_y,deg = 1)
    rad = np.arctan(p[0])
    
    return rad
def draw(im,line,idx,show = False):
    '''
    Generate the segmentation label according to json annotation
    '''
    line_x = line[::2]
    line_y = line[1::2]
    pt0 = (int(line_x[0]),int(line_y[0]))
    if show:
        cv2.putText(im,str(idx),(int(line_x[len(line_x) // 2]),int(line_y[len(line_x) // 2]) - 20),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = idx * 60

    for i in range(len(line_x)-1):
        cv2.line(im,pt0,(int(line_x[i+1]),int(line_y[i+1])),(idx * 60,),thickness = 6 )    # thickness = i//4 +1
        pt0 = (int(line_x[i+1]),int(line_y[i+1]))
    # cv2.imshow('im', cv2.resize(im,dsize=None,fx=0.5,fy=0.5))
    # cv2.waitKey(0)


def get_tusimple_list(root, label_list):
    '''
    Get all the files' names from the json annotation
    '''
    label_json_all = []
    for l in label_list:
        l = os.path.join(root,l)
        label_json = [json.loads(line) for line in open(l).readlines()]
        label_json_all += label_json
    names = [l['raw_file'] for l in label_json_all]
    h_samples = [np.array(l['h_samples']) for l in label_json_all]
    lanes = [np.array(l['lanes']) for l in label_json_all]

    line_txt = []
    for i in range(len(lanes)):
        line_txt_i = []
        for j in range(len(lanes[i])):
            if np.all(lanes[i][j] == -2):
                continue
            valid = lanes[i][j] != -2
            line_txt_tmp = [None]*(len(h_samples[i][valid])+len(lanes[i][j][valid]))
            line_txt_tmp[::2] = list(map(str,lanes[i][j][valid]))
            line_txt_tmp[1::2] = list(map(str,h_samples[i][valid]))
            line_txt_i.append(line_txt_tmp)
        line_txt.append(line_txt_i)

    return names,line_txt

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


def generate_segmentation_and_train_list(root,filedir):    #  , line_txt, names
    """
    The lane annotations of the Tusimple dataset is not strictly in order, so we need to find out the correct lane order for segmentation.
    We use the same definition as CULane, in which the four lanes from left to right are represented as 1,2,3,4 in segentation label respectively.
    """
    # filedir='valid'

    train_gt_fp = open(os.path.join(root,filedir,'train_gt.txt'),'w')

    fileList = os.listdir(os.path.join(root, filedir ,'images'))
    for idx,file in enumerate(fileList[:200]):
        # file = '70ed65f1ba25688c3bfccf3ea37e6416.jpg'
        jsonname = file.replace('.jpg','.lines.json')
        if not os.path.isfile(os.path.join(root,filedir,'labels',jsonname)):
            print(os.path.join(root, filedir,'labels', jsonname))
            continue

        tmp_line = get_CurveLanes_point(os.path.join(root,filedir,'labels',jsonname))
        if tmp_line ==[]:
            continue

        lines = []
        for j in range(len(tmp_line)):
            lines.append(list(map(float,tmp_line[j])))
        
        ks = np.array([calc_k(line) for line in lines])             # get the direction of each lane

        k_neg = ks[ks<0].copy()
        k_pos = ks[ks>0].copy()
        k_neg = k_neg[k_neg != -10]                                      # -10 means the lane is too short and is discarded
        k_pos = k_pos[k_pos != -10]
        k_neg.sort()
        k_pos.sort()
        k_neg = list(k_neg)
        k_pos = list(k_pos)
        kn_l = len(k_neg)
        kp_l = len(k_pos)

        ke = 0.15

        # 2条线隔的太近删除靠外侧一条.
        if kn_l>=2:
            for i in range(kn_l-1):
                p1= kn_l -1 -i
                p2 = kn_l -2 -i
                if abs(k_neg[p1] - k_neg[p2]) < ke:
                    k_neg.pop(p1)
        if kp_l >= 2:
            for i in range(kp_l-1):
                p1 = kp_l - 1 - i
                p2 = kp_l - 2 - i
                if abs(k_pos[p1] - k_pos[p2]) < ke:
                    k_pos.pop(p2)

        # imgname = '%06d.jpg'%i
        newimgpath = os.path.join(root, filedir,'images',file)
        # shutil.copy(os.path.join(root,names[i]),newimgpath)
        img = cv2.imread(newimgpath)
        shape = img.shape
        label_path = file.replace('jpg','png')

        # label = np.zeros((1440,2560),dtype=np.uint8)
        label = np.zeros(shape[:2],dtype=np.uint8)
        # label = cv2.imread(newimgpath)

        bin_label = [0,0,0,0]
        if len(k_neg) == 1:                                           # for only one lane in the left
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label,lines[which_lane],2)
            bin_label[1] = 1
        elif len(k_neg) >= 2:                                         # for two lanes in the left

            which_lane = np.where(ks == k_neg[1])[0][0]
            draw(label,lines[which_lane],1)
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label,lines[which_lane],2)
            bin_label[0] = 1
            bin_label[1] = 1


        if len(k_pos) == 1:                                            # For the lanes in the right, the same logical is adopted.
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label,lines[which_lane],3)
            bin_label[2] = 1
        elif len(k_pos) >= 2:
            which_lane = np.where(ks == k_pos[len(k_pos)-1])[0][0]
            draw(label,lines[which_lane],3)
            which_lane = np.where(ks == k_pos[len(k_pos)-2])[0][0]
            draw(label,lines[which_lane],4)
            bin_label[2] = 1
            bin_label[3] = 1


        # print(names[i] + ' ' + label_path + ' '+' '.join(list(map(str,bin_label))) + '\n');quit()
        x = shape[0]/360
        if x == 4.0:
            y=0
        else:
            x=0
            y=185
        # print(x,y,shape)
        labelpath = os.path.join(root,filedir,'gt_image',label_path)
        label =label[int(8*x):,y:shape[1]-y]
        cv2.imwrite(labelpath,label)
        # img = img[int(8*x):,:]

        cv2.imwrite(os.path.join(root, filedir,'image',file), img[int(8*x):,y:shape[1]-y])

        train_gt_fp.write(newimgpath + ' ' + labelpath + ' '+' '.join(list(map(str,bin_label))) + '\n')
    train_gt_fp.close()

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--root',default='/home/wqg/data/LANE DETECTION CHALLENGE/train_set', required=True, help='The root of the Tusimple dataset')
    parser.add_argument('--root',default='/home/wqg/data/CurveLanes',  help='The root of the Tusimple dataset')
    return parser

if __name__ == "__main__":
    args = get_args().parse_args()

    # training set
    # names,line_txt = get_tusimple_list(args.root,  ['label_data_0601.json','label_data_0531.json','label_data_0313.json'])
    # generate segmentation and training list for training
    # os.makedirs(os.path.join(args.root, 'train' ,'image'), exist_ok=True)
    for file in ['train','valid']:
        os.makedirs(os.path.join(args.root, file ,'gt_image'), exist_ok=True)
        os.makedirs(os.path.join(args.root, file ,'image'), exist_ok=True)
        # os.makedirs(os.path.join(args.root, 'train' ,'gt_image'), exist_ok=True)

        generate_segmentation_and_train_list(args.root,file)

    # testing set
    # names,line_txt = get_tusimple_list(args.root, ['test_tasks_0627.json'])
    # generate testing set for testing
    # with open(os.path.join(args.root,'test.txt'),'w') as fp:
    #     for name in names:
    #         fp.write(name + '\n')

   # --root  /home/wqg/data/LANE_DETECTION_CHALLENGE/train_set