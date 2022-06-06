
import scipy.io as scio
import cv2




def read_M(path):


    dict_data = scio.loadmat(path)
    rgb_seg_vp = dict_data['rgb_seg_vp']
    # print(rgb_seg_vp)
    img = rgb_seg_vp[: , : ,:3]
    mask1 = rgb_seg_vp[: , : , 3]
    mask = rgb_seg_vp[: , : , 3]*10
    # img5 = rgb_seg_vp[: , : , 4]



    cv2.imshow('img',img)
    cv2.imshow('mask',mask)
    cv2.imshow('mask1',mask1)
    cv2.waitKey(0)



if __name__ =="__main__":
    path = '/home/wqg/data/VPGNet/scene_1/20160512_1331_31/000031.mat'
    read_M(path)