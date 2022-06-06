'''
将文字区域的外接矩形框绘制在图片上
'''
import os
import cv2
from tqdm import tqdm
import argparse

class ProcessTxt:
    def __init__(self,args):
        self.imgs_dir = args.imgs_dir
        self.label_dir = args.label_dir
        self.img_save_dir = args.img_save_dir
        self.mkdir_dir(self.img_save_dir)

    def mkdir_dir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            files = os.listdir(path)
            for file in files:
                file_path = os.path.join(path, file)
                os.remove(file_path)

    def process(self):
        imgs = os.listdir(self.imgs_dir)
        for i_img in tqdm(range(len(imgs))):
            img = imgs[i_img]
            img_path = os.path.join(self.imgs_dir, img)
            pic = cv2.imread(img_path)
            label_path = os.path.join(self.label_dir, img[:-4]+'.txt')
            if os.path.exists(label_path):
                lines = open(label_path, 'r', encoding='utf-8').readlines()
                point = None
                for line in lines:
                    point = list(map(int, line.strip().split(',')))

                # 矩形左上角和右上角的坐标，绘制一个绿色矩形
                ptLeftTop = (point[0], point[1])
                ptRightBottom = (point[2], point[3])
                point_color = (0, 255, 0)  # BGR
                thickness = 1
                lineType = 4
                cv2.rectangle(pic, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
                img_path_save = os.path.join(self.img_save_dir, img)
                cv2.imwrite(img_path_save, pic)


def parse_args():
    parser = argparse.ArgumentParser()
    part_ = '891'
    parser.add_argument('--imgs_dir',
                        default='/home/fuxueping/sdb/data/container/container/ori/'+part_+'/imageset',
                        help='图片文件所在文件夹')
    parser.add_argument('--label_dir',
                        default='/home/fuxueping/sdb/data/container/container/ori/'+part_+'/det_label',
                        help='字符识别标签文件所在文件夹'
                        )
    parser.add_argument('--img_save_dir',
                        default='/home/fuxueping/sdb/data/container/container/ori/'+part_+'/display_det_label',
                        help='画上标签的图片存储路径'
                        )

    return parser.parse_args()

def main(args):
    G = ProcessTxt(args)
    G.process()


if __name__ == '__main__':
    main(parse_args())