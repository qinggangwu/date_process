
"""
解析vpgnet数据,标签17类,大小 480 * 640
{1:3 ,2:1 ,3:3 ,4:4 ,5:2 ,6:4 ,7:2 ,8:3 ,9:6 ,10:5 ,11:5 ,12:5 ,13:5 ,14:0 ,15:0 ,16:5 ,17:0  }

### Lane and road markings (4th channel) ###
0	background               背景
1	lane_solid_white		 白实线
2	lane_broken_white		 白虚线
3	lane_double_white		 双白线
4	lane_solid_yellow 		 黄实线
5	lane_broken_yellow		 黄虚线
6	lane_double_yellow		 双黄线
7	lane_broken_blue		 蓝虚线(蓝色的线为公交车道线)
8	lane_slow 				 缓慢行驶线(类似于s形状白实线)
9	stop_line	 			 停止线
10	arrow_left  			 导流线 (左转)
11	arrow_right 			 导流线 (右转)
12	arrow_go_straight		 导流线 (直行)
13	arrow_u_turn			 导流线 (掉头)
14	speed_bump				 减速行驶线
15	crossWalk				 斑马线
16	safety_zone				 导流线(上下高速路口汇流导流线)
17	other_road_markings		 其他  (地面文字, 左转和直行合并地标,菱形地标,无法区分的导流线地标)

### Vanishing Points (5th channel) ###
0	none/background         无消失点
1	easy					明显消失点
2	hard					预测消失点


公交线路专用车道线的颜色为蓝色（韩国为蓝色，中国为黄色）
"""


import scipy.io as scio
import cv2


import scipy.io
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse


def walkThroughDataset(dataSetDir):
    markClass = [
        ### Lane and road markings (4th channel) ###
        'background',
        'lane_solid_white',
        'lane_broken_white',
        'lane_double_white',
        'lane_solid_yellow',
        'lane_broken_yellow',
        'lane_double_yellow',
        'lane_broken_blue',
        'lane_slow',
        'stop_line',
        'arrow_left',
        'arrow_right',
        'arrow_go_straight',
        'arrow_u_turn',
        'speed_bump',
        'crossWalk',
        'safety_zone',
        'other_road_markings']

    vpClass = [
        ### Vanishing Points (5th channel) ###
        'background',
        'easy',
        'hard',
    ]

    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dataSetDir):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames if os.path.splitext(file)[-1] == '.mat' ]

    """
    /home/wqg/data/VPGNet/scene_4/20160518_2015_20/000121.mat
    /home/wqg/data/VPGNet/scene_4/20160518_1953_48/000151.mat
    /home/wqg/data/VPGNet/scene_4/20160805_2122_27/000271.mat
    """


    colorMapMat = np.zeros((18, 3), dtype=np.uint8)
    # laneColor = np.array([0, 255, 0], dtype=np.uint8)
    # vpColorMapMat = np.array([[0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    for i in range(0, len(markClass)):
        if i != 0:
            colorMapMat[i] = np.random.randint(0, 255, dtype=np.uint8, size=3)

    # colorMapMat = [(102, 0, 204), (0, 0,142), (119,11,32),(70,130,180), (220, 20,60), (180, 165, 180),(220, 220,   0),(178, 132, 190) ]

    for imageFile in tqdm(listOfFiles):
        data = scipy.io.loadmat(imageFile)
        rgb_seg_vp = data['rgb_seg_vp']
        rgb = rgb_seg_vp[:, :, 0: 3]
        seg = rgb_seg_vp[:, :, 3]

        point = np.where(seg == 16)
        if len(point[0]) == 0:
            continue

        seg = change_label(seg)
        # vp = rgb_seg_vp[:, :, 4]
        img_bgr = rgb[:, :, :: -1]
        segImage = colorMapMat[seg]
        # x = np.nonzero(vp)[1]
        # y = np.nonzero(vp)[0]
        # if x.size > 0 and y.size > 0:
        #     vpPoint = (x[0], y[0])
        #     vpLevel = vp[(y, x)]
        # else:
        #     vpPoint = (-1, -1)

        # if vpPoint[0] > 1 and vpPoint[1] > 1:
        #     c = vpColorMapMat[vpLevel - 1][0].tolist()
        #     cv2.circle(segImage, vpPoint, 15, (c[0], c[1], c[2]), thickness=-1, lineType=8)


        # cv2.imshow('VPGNet Dataset img_bgr', img_bgr)
        # cv2.imshow('VPGNet Dataset segImage', segImage)
        # cv2.waitKey(0)

        res = cv2.addWeighted(img_bgr, 1, segImage, 0.7, 0)
        cv2.imshow('VPGNet Dataset Quick Inspector', res)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
    cv2.destroyWindow('VPGNet Dataset Quick Inspector')


def change_label(seg):
    change_dict = {1:3, 2:1, 5:2, 6:4, 7:2, 8:3, 9:6, 10:5, 11:5, 12:5, 13:5, 14:0, 15:0, 16:5, 17:0}

    for orlabe in change_dict:
        value = change_dict[orlabe]

        seg[np.where(seg == orlabe)] = value
    return seg



def parse_args():
    parser = argparse.ArgumentParser(
        description='VPGNet Dataset Quick Inspector')
    parser.add_argument('--rootDir', type=str, default=r'/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/data/VPGNet',
                        help='root directory (default: D:\\VPGNet-DB-5ch)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    walkThroughDataset(args.rootDir)

"""
def read_M(path):


    dict_data = scio.loadmat(path)
    rgb_seg_vp = dict_data['rgb_seg_vp']
    # print(rgb_seg_vp)
    img = rgb_seg_vp[: , : ,:3]


    mask0 = rgb_seg_vp[: , : , 0]
    mask1 = rgb_seg_vp[: , : , 1]
    mask2 = rgb_seg_vp[: , : , 2]
    # mask3 = rgb_seg_vp[: , : , 3]
    # mask4 = rgb_seg_vp[: , : , 4]
    mask = rgb_seg_vp[: , : , 3]*10
    # img5 = rgb_seg_vp[: , : , 4]
    # img5 = rgb_seg_vp[: , : , 4]

    # for i in mask2:
    #     print(i)
    # print('end');quit()



    cv2.imshow('img',img)
    cv2.imshow('mask',mask)
    # cv2.imshow('mask0',mask0)
    # cv2.imshow('mask1',mask1)
    cv2.imshow('mask2',mask2)
    # cv2.imshow('mask3',mask3)
    # cv2.imshow('mask4',mask4)
    cv2.waitKey(0)


if __name__ =="__main__":
    path = '/home/wqg/data/VPGNet/scene_1/20160512_1331_31/000061.mat'
    read_M(path)

"""
