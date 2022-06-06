'''
将文字区域的外接矩形框绘制在图片上,txt上的标签绘制到图片上
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
                if os.path.splitext(file)[-1] in ['.jpg','.png']:
                    file_path = os.path.join(path, file)
                    os.remove(file_path)
                

    def process(self):
        num = 0 
        imgs = os.listdir(self.imgs_dir)
        for i_img in tqdm(range(len(imgs))):
            img = imgs[i_img]
            img_path = os.path.join(self.imgs_dir, img)
            pic = cv2.imread(img_path)
            try:
                h,w,c = pic.shape
            except:
                print(img_path)
                continue
                
            label_path = os.path.join(self.label_dir, img[:-4]+'.txt')
            # head_s = 0
            # body_s = h * w
            if os.path.exists(label_path):
                lines = open(label_path, 'r', encoding='utf-8').readlines()
                point = None
                label = None
                for line in lines:
                    # point = list(map(int, line.strip().split(' ')))
                    label = line.strip().split(' ')[0]
                    point = list(map(float,line.strip().split(' ')[1:])) #读取中点，w，h
                    center_x = point[0]*w
                    center_y = point[1]*h
                    w_obj = point[2]*w
                    h_obj = point[3]*h
                    x_tl = int(center_x-0.5*w_obj)
                    y_tl = int(center_y-0.5*h_obj)
                    x_br = int(center_x+0.5*w_obj)
                    y_br = int(center_y+0.5*h_obj)

                    # 矩形左上角和右下角的坐标，绘制一个绿色矩形
                    ptLeftTop = (x_tl, y_tl)
                    ptRightBottom = (x_br, y_br)
                    point_color = (0, 255, 0)  # BGR
                    thickness = 1
                    lineType = 4
                    cv2.rectangle(pic, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

                    cv2.putText(pic, label, (x_tl, y_tl), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness)
                        # head_s_ = int(w_obj * h_obj)
                        # head_s = max(head_s, head_s_)
                    # else:
                    #     body_s_ = int(w_obj * h_obj)
                    #     body_s = min(body_s, body_s_)
                    #     cv2.putText(pic, 'body', (x_tl, y_tl), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), thickness)
                # if body_s < head_s:
                img_path_save = os.path.join(self.img_save_dir, img)
                # print(self.img_save_dir)
                cv2.imwrite(img_path_save, pic)

            else:
                num+=1
        print("无字符外框图片数量 :",num)


def parse_args(part_):
    parser = argparse.ArgumentParser()
    # part_ = '1658'
    pathdir = '/home/jovyan/data-vol-1/wqg/container/obj_det/'
    parser.add_argument('--imgs_dir',
                        default=pathdir +part_+'/images',
                        help='图片文件所在文件夹')
    parser.add_argument('--label_dir',
                        default=pathdir+part_+ '/labels',
                        help='字符识别标签文件所在文件夹'
                        )
    parser.add_argument('--img_save_dir',
                        default=pathdir+part_+'/display_det_label',
                        help='画上标签的图片存储路径'
                        )
    print(part_)
    return parser.parse_args()

def main(args):
    G = ProcessTxt(args)
    G.process()


if __name__ == '__main__':
#     part_all = ['950','951','952','953','954','955','956','963','964','965','966','967','973','974','975','998','1793', '1792', '1791']
#     part_all = ['1793', '1792', '1791']
    part_all =['1777', '1722', '972', '971', '970', '969', '968', '962', '961', '960', '959', '958', '940', '939', '938', '937', '936', '935', '934', '933', '928', '925', '924', '923', '922', '892', '891', '630','950','951','952','953','954','955','956','963','964','965','966','967','973','974','975','998','1793', '1792', '1791','1793', '1792', '1791']
    for part_ in part_all:
        main(parse_args(part_))
#     main(parse_args())