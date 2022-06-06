import os
import random
import shutil
import argparse
from tqdm import tqdm

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class MovePic():
    def __init__(self,arg):
        self.dir = arg.dirpath
        self.partList = arg.partList
        self.randomSendNum = random.seed(10)
        self.val_List = []
        self.all_List = []
        self.val_num = 700  # 验证集图片数量

    # 获取所有图片文件,创建all_List 和val_List
    def get_all_list(self,n=700):
        self.all_List = []
        savedir = os.path.join(self.dir, "ll_seg_annotations")

        make_dir(os.path.join(savedir, "train"))
        make_dir(os.path.join(savedir, "val"))

        for part_ in self.partList:
            for f in os.listdir(os.path.join(self.dir, part_, 'imageset')):
                if os.path.splitext(f)[-1] == '.jpg':
                    self.all_List.append(os.path.join(self.dir, part_, 'imageset', f))
        random.seed(10)
        self.val_List = random.choices(self.all_List, k=n)    # k=n  随机抽取n个数据作为验证集
        self.val_List = list(set(self.val_List))
        print("验证集数据数量: %d"%len(self.val_List))

    def comp_file(self):
        pass

    # 移动原始图片训练数据
    def move_image_data(self):

        # 判断是否存在val_list  不存在就索引所有文件
        if self.val_List == []:
            self.get_all_list(self.val_num)

        # 创建文件夹
        make_dir(os.path.join(self.dir, 'images', 'train'))
        make_dir(os.path.join(self.dir, 'images', 'val'))
        print("移动 images 文件夹")
        for img in tqdm(self.all_List):
            if img in self.val_List:
                shutil.copy(img, os.path.join(self.dir, 'images', 'val'))
            else:
                shutil.copy(img, os.path.join(self.dir, 'images', 'train'))

    # 移动车道分割线训练数据
    def move_llseg_data(self):
        # 判断是否存在val_list  不存在就索引所有文件
        if self.val_List == []:
            self.get_all_list(self.val_num)

        # 创建文件夹
        make_dir(os.path.join(self.dir,'ll_seg_annotations','train'))
        make_dir(os.path.join(self.dir,'ll_seg_annotations','val'))
        print("移动 ll_seg_annotations 文件夹")
        for img in tqdm(self.all_List):
            new_img = img.replace('imageset','line_mask').replace('jpg','png')
            try:
                if img in self.val_List:
                    shutil.copy(new_img,os.path.join(self.dir,'ll_seg_annotations','val'))
                else:
                    shutil.copy(new_img, os.path.join(self.dir, 'll_seg_annotations', 'train'))
            except:
                print(new_img)

    # 移动 车辆检测 训练数据
    def move_det_data(self):

        if self.val_List == []:
            self.get_all_list(self.val_num)

        make_dir(os.path.join(self.dir, 'det_annotations', 'train'))
        make_dir(os.path.join(self.dir, 'det_annotations', 'val'))
        print('移动 da_seg_annotations 文件夹' )
        for img in tqdm(self.all_List):
            new_img = img.replace('imageset', 'det_annotations').replace('jpg', 'json')
            try:
                if img in self.val_List:
                    shutil.copy(new_img, os.path.join(self.dir, 'det_annotations', 'val'))
                else:
                    shutil.copy(new_img, os.path.join(self.dir, 'det_annotations', 'train'))
            except:
                print(new_img)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirpath',
                        default= '/home/wqg/data/公司数据/car_line/',
                        type= str,
                        help='图片文件所在文件夹的根目录')
    parser.add_argument('--partList',
                        default=['657'],
                        # default=['657', '658', '656', '655', '654', '653', '652'],
                        type= list,
                        help='图片文件夹名称的列表'
                        )

    # parser.add_argument('--save_clip_det_region',
    #                     default=pathdir + 'obj_det/' + part_ + '/images',
    #                     help='保存做字符区域分割的标签的文件路径'
    #                     )

    return parser.parse_args()

def main():
    arg = parse_args()
    G = MovePic(arg)
    # G.move_image_data()
    # G.move_llseg_data()
    G.move_det_data()

if __name__ == "__main__":
    main()