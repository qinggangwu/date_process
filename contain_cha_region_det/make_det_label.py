'''
将detbox转换为可以用于yolov5训练的标签
'''
import argparse
import os
import shutil
import cv2
from tqdm import tqdm

class ProcessTxt:
    def __init__(self, args):
        self.label_dir = args.label_dir
        self.save_label_dir = args.save_label_dir
        txts = os.listdir(self.save_label_dir)
        for txt in txts:
            save_txt_path = os.path.join(self.save_label_dir, txt)
            os.remove(save_txt_path)
        self.img_dir = args.img_dir
        self.img_save = args.img_save
        self.save_flag = args.save_flag
        save_imgs = os.listdir(self.img_save)
        for img_save in save_imgs:
            save_img_path = os.path.join(self.img_save, img_save)
            os.remove(save_img_path)

    def process(self):
        txts = os.listdir(self.label_dir)
        for i_txt in tqdm(range(len(txts))):
            txt = txts[i_txt]
            txt_path = os.path.join(self.label_dir, txt)
            part_name = os.path.splitext(txt)[0]
            img_path = os.path.join(self.img_dir, part_name+'.jpg')
            pic = cv2.imread(img_path)
            pic_h, pic_w = pic.shape[:2]
            all_lines = open(txt_path, 'r', encoding='utf-8').readlines()
            if len(all_lines) != 0:
                save_txt = os.path.join(self.save_label_dir, txt)
                f_save = open(save_txt, 'w', encoding='utf-8')
                for line in all_lines:
                    point = line.strip().split(',')
                    #class_index center_x center_y width height (其中center_x center_y width height都是归一化的数值：)
                    #center_x/ori_pic_w
                    #center_y/ori_pic_h
                    #width   /ori_pic_w
                    #height  /ori_pic_h
                    ltx, lty, rbx, rby = int(point[0]),int(point[1]),int(point[2]),int(point[3])
                    width = rbx - ltx
                    height = rby - lty
                    center_x = ltx+int(0.5*width)
                    center_y = lty+int(0.5*height)


                    # point_color = (0, 0, 0)  # BGR
                    # thickness = 3
                    # lineType = 4
                    # cv2.line(pic, (ltx, center_y), (rbx, center_y), point_color, thickness, lineType)
                    # cv2.line(pic, (center_x, lty), (center_x, rby), point_color, thickness, lineType)

                    # ptLeftTop = (ltx, lty)
                    # ptRightBottom = (rbx, rby)
                    # point_color = (0, 255, 0)  # BGR
                    # thickness = 1
                    # lineType = 4
                    # cv2.rectangle(pic, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

                    # point_size = 2
                    # point_color = (0, 0, 255)  # BGR
                    # thickness = 6  # 可以为 0 、4、8
                    # # 要画的点的坐标
                    # cv2.circle(pic, (center_x, center_y), point_size, point_color, thickness)

                    new_line = '0'+' '+str(format(center_x/pic_w, '.6f'))+' '+\
                               str(format(center_y/pic_h, '.6f'))+' '+\
                               str(format(width/pic_w, '.6f'))+' '+\
                               str(format(height/pic_h, '.6f'))

                    f_save.write(new_line)
                f_save.close()
                if self.save_flag == True:
                    save_img_path = os.path.join(self.img_save, part_name+'.jpg')
                    cv2.imwrite(save_img_path, pic)





def parse_args():
    parser = argparse.ArgumentParser()
    # part_ = '940'
    parser.add_argument('--img_dir',
                        default='/home/fuxueping/sdb/data/container/container_obj_det/single_cha/2020_5_7/train/images',
                        help='图片所在位置')
    parser.add_argument('--label_dir',
                        default='/home/fuxueping/sdb/data/container/container_obj_det/single_cha/2020_5_7/train/labels',
                        help='标签所在位置')
    parser.add_argument('--save_label_dir',
                        default='/home/fuxueping/sdb/data/container/container_obj_det/single_cha/2020_5_7/train/labels',
                        help='生成标签存储位置'
                        )
    parser.add_argument('--img_save',
                        default='/home/fuxueping/sdb/data/container/container_obj_det/single_cha/2020_5_7/train/temp',
                        help='保存标签标注在图片上的结果的图片路径')
    parser.add_argument('--save_flag',
                        default=True,
                        help='是否保存标签标注在图片上的结果')

    return parser.parse_args()

def main(args):
    G = ProcessTxt(args)
    G.process()


if __name__ == '__main__':
    main(parse_args())