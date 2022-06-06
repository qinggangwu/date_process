'''
集装箱字符区域检测第一步：（大块区域的检测）
1）并挑选出有多个目标检测区域的图片，保存在文件夹中，不做任何处理
2）对非多目标检测区域的图片做以下处理：
  得出做目标检测区域的标签
  裁剪出目标检测标签框选出的区域
  得出用于文字区域分割的标签
'''
import os
import cv2
from tqdm import tqdm
from math import ceil
import argparse
import shutil



class ProcessTxt:
    def __init__(self, args):
        self.img_dir = args.img_dir
        self.label_dir = args.label_dir

        self.save_clip_region = args.save_clip_region
        self.mkdir_dir(self.save_clip_region)

        self.save_seg_label = args.save_seg_label
        self.mkdir_dir(self.save_seg_label)

        self.save_det_label = args.save_det_label
        self.mkdir_dir(self.save_det_label)

        self.save_special_dir = args.save_special_dir
        self.mkdir_dir(self.save_special_dir)

    def mkdir_dir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            files = os.listdir(path)
            for file in files:
                file_path = os.path.join(path, file)
                os.remove(file_path)


    def process(self):
        txt_paths = os.listdir(self.label_dir)
        for i_txt_path in tqdm(range(len(txt_paths))):
            txt_path = txt_paths[i_txt_path]
            # print(txt_path)
            txt_ab_path = os.path.join(self.label_dir, txt_path)
            img_path = os.path.join(self.img_dir, txt_path[:-4]+'.jpg')
            img = cv2.imread(img_path)
            # img_h, img_w = img.shape[:2]
            h_img, w_img, c = img.shape
            all_lines = open(txt_ab_path, 'r', encoding='utf-8').readlines()
            min_x = w_img
            min_y = h_img
            max_x = 0
            max_y = 0
            if len(all_lines) <= 4:
                for line in all_lines:
                    if len(line.strip()) != 0:
                        line_point = line.strip().split('\t')[0].split(',') #得到所有坐标点
                        line_point_x = [int(line_point[0]), int(line_point[2]),int(line_point[4]),int(line_point[6])]
                        line_point_y = [int(line_point[1]), int(line_point[3]),int(line_point[5]),int(line_point[7])]
                        line_point_x.sort()
                        line_point_y.sort()
                        min_x = min(line_point_x[0], min_x)
                        min_y = min(line_point_y[0], min_y)
                        max_x = max(line_point_x[3], max_x)
                        max_y = max(line_point_y[3], max_y)

                w = max_x - min_x
                h = max_y - min_y
                w_ratio, h_ratio = 0.10, 0.10
                if w > h:
                    w_ratio = 0.05
                else:
                    h_ratio = 0.05

                # for train
                x1 = int(max(0, min_x - ceil(w * w_ratio)))
                y1 = int(max(0, min_y - ceil(h * h_ratio)))
                x2 = int(min(w_img, max_x + ceil(w * w_ratio)))
                y2 = int(min(h_img, max_y + ceil(h * h_ratio)))

                new_seg_label = os.path.join(self.save_seg_label, txt_path[:-4]+'.txt')
                f_save_seg = open(new_seg_label, 'w', encoding='utf-8')
                for line in all_lines:
                    if len(line.strip()) != 0:
                        line_point = list(map(int, line.strip().split(',')[0:8]))  #得到所有坐标点
                        label = line.strip().split(',')[8]
                        line_point_str = str(line_point[0] - x1)+','+str(line_point[1] - y1)+',' +\
                                         str(line_point[2] - x1)+','+str(line_point[3] - y1)+',' +\
                                         str(line_point[4] - x1)+','+str(line_point[5] - y1)+',' +\
                                         str(line_point[6] - x1)+','+str(line_point[7] - y1)+','+ label+'\n'

                        f_save_seg.write(line_point_str)

                f_save_seg.close()


                det_img_save = os.path.join(self.save_clip_region, txt_path[:-4]+'.jpg')
                det_img = img[y1:y2, x1:x2, :]
                cv2.imwrite(det_img_save, det_img)


                # rec_point_str = str(min_x)+','+str(min_y)+','+str(max_x)+','+str(max_y)+'\n'
                rec_point_str = str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + '\n'
                save_txt = os.path.join(self.save_det_label, txt_path)
                f_save = open(save_txt, 'w', encoding='utf-8')
                f_save.write(rec_point_str)
                f_save.close()

            else:
                save_special_path = os.path.join(self.save_special_dir, txt_path[:-4]+'.jpg')
                shutil.copy(img_path, save_special_path)



def parse_args():
    parser = argparse.ArgumentParser()
    part_ = '630'
    parser.add_argument('--img_dir',
                        default='/home/fuxueping/sdb/data/container/container/ori/'+part_+'/imageset',
                        help='图片文件所在文件夹')
    parser.add_argument('--label_dir',
                        default='/home/fuxueping/sdb/data/container/container/ori/'+part_+'/train_labels',
                        help='字符识别标签文件所在文件夹'
                        )
    parser.add_argument('--save_clip_region',
                        default='/home/fuxueping/sdb/data/container/container/ori/'+part_+'/clip_region',
                        help='保存抠取出的目标区域图片'
                        )
    parser.add_argument('--save_seg_label',
                        default='/home/fuxueping/sdb/data/container/container/ori/'+part_+'/seg_label',
                        help='保存做字符区域分割的标签的文件路径'
                        )
    parser.add_argument('--save_det_label',
                        default='/home/fuxueping/sdb/data/container/container/ori/' + part_ + '/det_label',
                        help='保存做字符区域目标检测标签的文件路径'
                        )
    parser.add_argument('--save_special_dir',
                        default='/home/fuxueping/sdb/data/container/container/ori/'+part_+'/special_img',
                        help='保存存在多车辆的图片路径')

    return parser.parse_args()

def main(args):
    G = ProcessTxt(args)
    G.process()


if __name__ == '__main__':
    main(parse_args())