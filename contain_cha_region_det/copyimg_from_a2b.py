"""
这块代码有功能：1）将a文件夹中的图片拷贝到b文件夹
             2）判断两张图片内容是否一样
"""
import os
import shutil
import cv2
import numpy as np


def mkdir_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            os.remove(file_path)


if __name__=='__main__':
    a_dir = '/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/container_obj_det/1722/1722/images'
    labels_a = '/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/container_obj_det/1722/1722/labels'
    b_dir = '/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/test_fromXJJ/错误'

    label_dir = '/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/container_obj_det/1722/1722/labels'
    save_dir = '/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/test_fromXJJ/temp/images'
    mkdir_dir(save_dir)
    save_dir_txt = '/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/test_fromXJJ/temp/labels'
    mkdir_dir(save_dir)

    files = os.listdir(b_dir)
    # for file in files:
    #     img_dir_a = os.path.join(a_dir, file)
    #     if os.path.exists(img_dir_a):
    #         save_path = os.path.join(save_dir, file)
    #         shutil.copy(img_dir_a, save_path)

    a_files = os.listdir(a_dir)
    for file in files:
        img_dir_b = os.path.join(b_dir, file)
        b_pic = cv2.imread(img_dir_b)


        for file_a in a_files:
            img_dir_a = os.path.join(a_dir, file_a)
            a_pic = cv2.imread(img_dir_a)

            difference = cv2.subtract(a_pic, b_pic)
            # print(difference)
            result = not np.any(difference)  # if difference is all zeros it will return False

            if result is True:
                print("两张图片一样")
                # save_path_a = os.path.join(save_dir, file[:-4]+'_ori.jpg')
                save_path_a = os.path.join(save_dir, file_a)
                cv2.imwrite(save_path_a, a_pic)

                ori_txt_path = os.path.join(labels_a, file_a[:-4]+'.txt')
                save_txt_a = os.path.join(save_dir_txt, file_a[:-4]+'.txt')
                shutil.copy(ori_txt_path, save_txt_a)

                # save_path_b = os.path.join(save_dir, file)
                # cv2.imwrite(save_path_b, b_pic)
            # else:
            #     print("两张图片不一样")

