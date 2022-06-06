"""
move_img

"""


import os.path as ops
import os
import shutil
import argparse
from tqdm import tqdm



def init_args():
    """
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='The origin path of unzipped tusimple dataset',
                        default='/home/wqg/data/lane')
    parser.add_argument('--fileList', type=list, help='Tag for validation set', default=['CULane'])
    # parser.add_argument('--fileList', type=list, help='Tag for validation set', default=['tusimple'])
    # parser.add_argument('--test', type=bool, help='Tag for validation set', default=True)

    return parser.parse_args()

def process_tusimple_dataset(src_dir , flieList):
    """

    :param src_dir:
    :param flieList:
    :return:
    """

    for filename in flieList:
        os.makedirs(ops.join(src_dir,filename,'val','gt_image'), exist_ok=True)
        os.makedirs(ops.join(src_dir,filename,'val','gt_binary_image'), exist_ok=True)
        os.makedirs(ops.join(src_dir,filename,'val','gt_instance_image'), exist_ok=True)

        for imgname in os.listdir(ops.join(src_dir,filename,'train','gt_image')):
            maskimgname = imgname.replace('jpg','png')
            if int(imgname[:-4]) % 5 == 0:
                # print(ops.join(src_dir,filename,'train','gt_image',imgname), ops.join(src_dir,filename,'val','gt_image',imgname))
                shutil.move(ops.join(src_dir,filename,'train','gt_image',imgname), ops.join(src_dir,filename,'val','gt_image',imgname))
                shutil.move(ops.join(src_dir,filename,'train','gt_binary_image',maskimgname), ops.join(src_dir,filename,'val','gt_binary_image',maskimgname))
                shutil.move(ops.join(src_dir,filename,'train','gt_instance_image',maskimgname), ops.join(src_dir,filename,'val','gt_instance_image',maskimgname))

def get_txt(src_dir,fileList):
    trainList = []
    valList =[]
    root_dir = '/home/wqg/data/lane'
    for file in fileList:
        train_feList = os.listdir(ops.join(root_dir,file,'train','gt_binary_image'))
        print(ops.join(root_dir,file,'train','gt_binary_image'))
        for imgname in tqdm(train_feList):

            image = ops.join(src_dir,file,'train','gt_image',imgname.replace('png','jpg'))
            binary = ops.join(src_dir,file,'train','gt_binary_image',imgname)
            instance = ops.join(src_dir,file,'train','gt_instance_image',imgname)
            info = "{} {} {}\n".format(image,binary,instance)
            trainList.append(info)

        val_feList = os.listdir(ops.join(root_dir, file, 'val', 'gt_binary_image'))
        print(ops.join(root_dir, file, 'val', 'gt_binary_image'))
        for imgname in tqdm(val_feList):
            image = ops.join(src_dir, file, 'val', 'gt_image', imgname.replace('png','jpg'))
            binary = ops.join(src_dir, file, 'val', 'gt_binary_image', imgname)
            instance = ops.join(src_dir, file, 'val', 'gt_instance_image', imgname)
            info = "{} {} {}\n".format(image, binary, instance)
            valList.append(info)

    with open(ops.join(root_dir,'train.txt') , 'w') as tr:
        tr.writelines(trainList)

    with open(ops.join(root_dir,'val.txt') , 'w') as te:
        te.writelines(valList)

    print("end")

if __name__ == '__main__':
    args = init_args()
    # process_tusimple_dataset(args.src_dir, args.fileList)
    # get_txt('/home/jovyan/data-vol-1/wqg/lane',['tusimple','CurveLanes'])
    # get_txt('/home/jovyan/data-vol-1/wqg/lane',['tusimple','CurveLanes','CULane'])

    dir_root = '/home/wqg/data/lane/CurveLanes/val/gt_image/'
    imgList = os.listdir(dir_root)
    for name in imgList:
        newname = '1'+name[1:]
        os.rename(dir_root+name,dir_root+newname)



