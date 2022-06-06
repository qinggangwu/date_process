'''
集装箱字符区域检测第二步：（单个字符区域的检测和分类）
1）将大块区域分割出来；
2）找到大块区域内的字符标签
3）将忽略的区域用灰度图遮掩起来
4）生成最终的标签
'''
import os
import cv2
from tqdm import tqdm
from math import ceil
import argparse
import numpy as np
import copy
import shutil



class ProcessTxt:
    def __init__(self, args):
        self.img_dir = args.img_dir
        self.label_dir = args.label_dir

        self.save_clip_region = args.save_clip_region
        self.mkdir_dir(self.save_clip_region)

        self.save_seg_label = args.save_seg_label
        self.mkdir_dir(self.save_seg_label)
        
        self.save_clip_det_region = args.save_clip_det_region
        self.mkdir_dir(self.save_clip_det_region)

        self.dict_labels = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
            'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':18,
         'J':19,'K':20,'L':21,'M':22,'N':23,'O':24,'P':25,'Q':26,'R':27,'S':28,'T':29,'U':30,'V':31,
         'W':32,'X':33,'Y':34,'Z':35}

        # self.save_det_label = args.save_det_label
        # self.mkdir_dir(self.save_det_label)

        # self.save_special_dir = args.save_special_dir
        # self.mkdir_dir(self.save_special_dir)

    def mkdir_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            files = os.listdir(path)
            for file in files:
                if os.path.splitext(file)[-1] in ['.jpg','.png']:
                    file_path = os.path.join(path, file)
                    os.remove(file_path)

    def change_point(self, small_box, large_box, large_box_attri):
        """
        将大区域中小目标找出来，并将标签制作成归一化的形式
        :param small_box: 小目标的box
        :param large_box: 大目标的box
        :param large_box_attri: 大目标的属性
        :return:
        """
        x0 = small_box[0]
        y0 = small_box[1]
        x1 = small_box[2]
        y1 = small_box[3]
        w_box_ = x1 - x0
        h_box_ = y1 - y0
        center_x = x0 + 0.5 * w_box_
        center_y = y0 + 0.5 * h_box_

        if (large_box[0] < center_x < large_box[2]) and (large_box[1] < center_y < large_box[3]):
            x0_large = large_box[0]
            y0_large = large_box[1]
            # x1_large = large_box[2]
            # y1_large = large_box[3]
            new_point = [x0 - x0_large,
                         y0 - y0_large,
                         x1 - x0_large,
                         y1 - y0_large]

            w_large = large_box_attri['w']
            h_large = large_box_attri['h']

            w_small_box = w_box_ * 1.0/w_large
            h_small_box = h_box_ * 1.0/h_large
            small_center_x = (new_point[0] + 0.5 * w_small_box*w_large)/w_large
            small_center_y = (new_point[1] + 0.5 * h_small_box*h_large)/h_large
            # return [small_center_x, small_center_y, w_small_box, h_small_box]
            # line_ = "%s %.6f %.6f %.6f %.6f\n" % (label_, center_x_norm, center_y_norm, w_box_norm, h_box_norm)
            line_ = "%.6f %.6f %.6f %.6f\n" % (small_center_x, small_center_y, w_small_box, h_small_box)
            return new_point, line_

        else:
            return None,None

    def display_label_on_img(self, label, box, pic):
        point_color = (0, 255, 0)  # BGR
        thickness = 4
        lineType = 4
        ptLeftTop = (int(box[0]), int(box[1]))
        ptRightBottom = (int(box[2]), int(box[3]))

        cv2.rectangle(pic, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        cv2.putText(pic, label, ptLeftTop, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness)

        cv2.imshow('result', pic)
        cv2.waitKey(1000)

    def process(self):
        txt_paths = os.listdir(self.label_dir)
        for i_txt_path in tqdm(range(len(txt_paths))):
            txt_path = txt_paths[i_txt_path]
            # print(txt_path)
            txt_ab_path = os.path.join(self.label_dir, txt_path)
            #-nfs-集装箱箱号识别数据-集装箱_OCR_label_20211013-阿曼-集装箱_OCR_label_20211013-阿曼-正确-20211002121224_Null_1.jpg
            part_img_name = txt_path[:-4]+'.jpg'
            img_path = os.path.join(self.img_dir, part_img_name)
            img = cv2.imread(img_path)

            # img_h, img_w = img.shape[:2]
#             h_img, w_img, c = img.shape
            try:
                h_img, w_img, c  = img.shape
            except:
                print(img_path)
                continue
            all_lines = open(txt_ab_path, 'r', encoding='utf-8').readlines()


            # fill_flag = False
            dict_point_label = {}
            dict_large_box ={}
            for line in all_lines:
                if len(line.strip()) != 0:
                    line_point = line.strip().split('\t')[0].split(',')  # 得到所有坐标点
                    label = line.strip().split('\t')[1].split('\n')[0]  # 得到标签
                    line_point_x = [int(line_point[0]), int(line_point[2]), int(line_point[4]),
                                    int(line_point[6])]
                    line_point_y = [int(line_point[1]), int(line_point[3]), int(line_point[5]),
                                    int(line_point[7])]
                    line_point_x.sort()
                    line_point_y.sort()

                    xlt = line_point_x[0]
                    ylt = line_point_y[0]
                    xrb = line_point_x[3]
                    yrb = line_point_y[3]
                    if label =='###' or label == 'unknown':
                        # fill_flag = True
                        mask_pts = np.array([[xlt, ylt], [xrb, ylt], [xrb, yrb], [xlt, yrb]])
                        cv2.fillPoly(img, [mask_pts], (114, 114, 114))
                        continue
                    elif '外框' in label : # label '集装箱外框'  or '集装箱号外框'
                        large_box = {}
                        w = xrb - xlt
                        h = yrb - ylt
                        w_ratio, h_ratio = 0.10, 0.10
                        if w > h:
                            w_ratio = 0.05
                        else:
                            h_ratio = 0.05

                        # for train
                        xlt = int(max(0, xlt - ceil(w * w_ratio)))
                        ylt = int(max(0, ylt - ceil(h * h_ratio)))
                        xrb = int(min(w_img, xrb + ceil(w * w_ratio)))
                        yrb = int(min(h_img, yrb + ceil(h * h_ratio)))
                        key_large = str(xlt) + ';' + str(ylt) + ';' + str(xrb) + ';' + str(yrb)
                        large_box['w'] = xrb - xlt
                        large_box['h'] = yrb - ylt
                        dict_large_box[key_large] = large_box
                    else:
                        if label not in dict_point_label.keys():
                            dict_point_label[label] = [[xlt, ylt, xrb, yrb]]
                        else:
                            value_point_label = dict_point_label[label]
                            value_point_label.append([xlt, ylt, xrb, yrb])
                            dict_point_label[label] = value_point_label


            # new_img = cv2.imread(img_path)
            # dict_large_box_smallobj = {}

            # 将填充为灰度图的图片再修正一下，防止有灰度图覆盖目标的情况；
            # for key_, value_ in dict_point_label.items():
            #     for box_ in value_:
            #         x0 = box_[0]
            #         y0 = box_[1]
            #         x1 = box_[2]
            #         y1 = box_[3]
            #
            #         if fill_flag == True:  # 将填充为灰度图的图片再修正一下，防止有灰度图覆盖目标的情况；
            #             img[y0:y1, x0:x1] = new_img[y0:y1, x0:x1]

            dict_large_box_smalls = {}
            for large_box, large_box_attri in dict_large_box.items():  # 判断字符目标在某个区间内
                large_box_int = list(map(int, large_box.split(';')))

                list_small_boxs = []
                # new_large_box_attri = copy.deepcopy(large_box_attri)

                all_lines = ''
                del_key_box = {}
                for key_, value_ in dict_point_label.items():
                    for box_ in value_:
                        # x0 = box_[0]
                        # y0 = box_[1]
                        # x1 = box_[2]
                        # y1 = box_[3]
                        # w_box_ = x1 - x0
                        # h_box_ = y1 - y0

                        new_point, point_obj_new = self.change_point(box_, large_box_int, large_box_attri)
                        if point_obj_new is not None:

                            # img_large_region = img[large_box_int[1]:large_box_int[3], large_box_int[0]:large_box_int[2]]
                            # large_region_h, large_region_w,large_region_c = img_large_region.shape
                            # point_obj_list = list(map(float, point_obj_new.lstrip('\n').split(' ')))
                            #
                            # point_obj_center_x = point_obj_list[0] * large_region_w
                            # point_obj_center_y = point_obj_list[1] * large_region_h
                            # point_obj_w = point_obj_list[2]*large_region_w
                            # point_obj_h = point_obj_list[3]*large_region_h
                            #
                            #
                            # new_point_ = [point_obj_center_x - 0.5 * point_obj_w,
                            #               point_obj_center_y - 0.5 * point_obj_h,
                            #               point_obj_center_x + 0.5 * point_obj_w,
                            #               point_obj_center_y + 0.5 * point_obj_h]
                            # self.display_label_on_img(key_, new_point_, img_large_region)
                            try:
                                all_lines += str(self.dict_labels[key_])+' '+point_obj_new
                            except:
                                print(key_, part_img_name)
                            # list_small_boxs.append(line)

                            if key_ in del_key_box.keys():
                                del_values = del_key_box[key_]
                                del_values.append(box_)
                                del_key_box[key_] = del_values

                            else:
                                del_key_box[key_] = [box_]

                # for __, value_1 in del_key_box.items(): #删除已经比对过的小目标
                #     value_all = dict_point_label[__]
                #     if len(value_1) == 1:
                #         del dict_point_label[__]
                #     else:
                #         for _ in value_1:
                #             if len(value_all) == 1:
                #                 del dict_point_label[__]
                #             else:
                #                 value_all.remove(_)
                #                 dict_point_label[__] = value_all
                    # pass

                dict_large_box_smalls[large_box] = all_lines

            num = 0
            for large_box_str, list_small_boxs in dict_large_box_smalls.items():
                large_box_int = list(map(int, large_box_str.split(';')))
                img_large_region = img[large_box_int[1]:large_box_int[3], large_box_int[0]:large_box_int[2]]
                save_img_path = os.path.join(self.save_clip_region, txt_path[:-4]+'_'+str(num)+'.jpg')
                save_txt_path = os.path.join(self.save_seg_label, txt_path[:-4]+'_'+str(num)+'.txt')
                cv2.imwrite(save_img_path, img_large_region)
                
                # 保存忽略框覆盖图片
                save_detimg_path = save_img_path = os.path.join(self.save_clip_det_region, part_img_name)
                cv2.imwrite(save_detimg_path, img)

                f_save_seg = open(save_txt_path, 'w', encoding='utf-8')
                f_save_seg.write(list_small_boxs)
                f_save_seg.close()
                num += 1



def parse_args(part_):
    parser = argparse.ArgumentParser()
    
#     part_ = '935_936'
    pathdir = '/home/jovyan/data-vol-1/wqg/container/'
     
    parser.add_argument('--img_dir',
                        # default='/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/container_det/'+part_+'/imageset',
                        default= pathdir + 'original_data/'+ part_+'/imageset',
                        help='图片文件所在文件夹')
    parser.add_argument('--label_dir',
                        # default='/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/container_obj_det/'+part_+'/labels_ori',
                        default= pathdir + 'original_data/' + part_+'/labels',
                        help='字符识别标签文件所在文件夹'
                        )
    parser.add_argument('--save_clip_region',
                        default=pathdir+'container_rec/'+part_+'/images',
                        help='保存抠取出的目标区域图片'
                        )
    parser.add_argument('--save_seg_label',
                        default=pathdir+'container_rec/'+part_+'/labels',
                        help='保存做字符区域分割的标签的文件路径'
                        )
    parser.add_argument('--save_clip_det_region',
                        default=pathdir+'obj_det/'+part_+'/images',
                        help='保存做字符区域分割的标签的文件路径'
                        )
    return parser.parse_args()

def main(args):
    G = ProcessTxt(args)
    G.process()


if __name__ == '__main__':
    part_all = ['950','951','952','953','954','955','956','963','964','965','966','967','973','974','975','998','1793', '1792', '1791']
    
#     part_all =['1777', '1722', '972', '971', '970', '969', '968', '962', '961', '960', '959', '958', '940', '939', '938', '937', '936', '935', '934', '933', '928', '925', '924', '923', '922', '892', '891', '630']
    
#     part_all = ['953']
    for part_ in part_all:
        main(parse_args(part_))