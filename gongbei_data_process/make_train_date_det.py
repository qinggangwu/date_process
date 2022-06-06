import os
import random
import shutil
from tqdm import tqdm


def mkdir_dir_(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        files = os.listdir(path)
        for file in files:
            if os.path.splitext(file)[-1] in ['.jpg', '.png']:
                file_path = os.path.join(path, file)
                os.remove(file_path)


# 移动图片,拆分训练集和测试集
def move_date_train_test():

    pathdir = '/home/wqg/data/gongbei/oringe/'
    part_all = ['1835', '1828', '1825', '1824', '1822', '1821', '1787', '1786', '1785']  # 1826号数据未标记.

    traindir = '/home/wqg/data/gongbei/car_det/'

    mkdir_dir_(os.path.join(traindir,"train","images"))
    mkdir_dir_(os.path.join(traindir,"train","labels"))

    mkdir_dir_(os.path.join(traindir, "valtest", "images"))
    mkdir_dir_(os.path.join(traindir, "valtest", "labels"))

    # 将所有的图片都移动到train 文件夹中
    for part_ in part_all:
        imgdir = os.path.join(pathdir,part_, 'images')
        txtdir = os.path.join(pathdir,part_, 'labels')
        print(part_)
        for imgfile in tqdm(os.listdir(imgdir)):
            txtfile = imgfile.replace("jpg",'txt')

            shutil.copy(os.path.join(imgdir,imgfile) ,os.path.join(traindir,'train',"images",imgfile))
            shutil.copy(os.path.join(txtdir,txtfile) ,os.path.join(traindir,'train','labels',txtfile))
    #
    print("数据总量 : " ,len( os.listdir(os.path.join(traindir,"train","images")) ))
    # print("数据总量 : " ,len( os.listdir(os.path.join(traindir,"valtest","images")) ))


    # 数据总量: 4775  4698


    valtestList = random.choices(os.listdir(os.path.join(traindir,"train","images")) ,k=800)
    for valimgfile in tqdm(valtestList):
        valtxtfile = valimgfile.replace("jpg", 'txt')

        # 移动图片
        try:
            shutil.move(os.path.join(traindir,"train","images",valimgfile),os.path.join(traindir,"valtest","images",valimgfile))
            shutil.move(os.path.join(traindir,"train","labels",valtxtfile),os.path.join(traindir,"valtest","labels",valtxtfile))
        # shutil.move(valtxtfile,valtxtfile_test)
        except:
            print(valimgfile)


if __name__ =="__main__":
    move_date_train_test()