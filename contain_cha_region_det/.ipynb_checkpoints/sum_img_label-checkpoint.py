'''
计算各个文件夹中的各个类别数量
'''
import os
import cv2
from tqdm import tqdm
import argparse
import math
import copy
from collections import Counter

class ProcessTxt:
    def __init__(self, args):
        self.file_names = args.file_names
        self.label_dir = args.label_dir
        self.classes_dict1 = {}



    def sum_label(self,label, classes_dict1):
        if label in classes_dict1.keys():
            value = classes_dict1[label]
            value += 1
            classes_dict1[label] = value
        else:
            print('label error:', label)

        return classes_dict1#label_0,label_1,label_2,label_3 = self.sum_label(label, label_0,label_1,label_2,label_3)



    def process(self):
        for file_name in self.file_names:
            classes_dict1 = {}

            with open("contain_cha_region_det/classes.name") as f:
                class_name = [ss[:-1] for ss in f.readlines()]
                # print(class_name)
                for idx, line in enumerate(class_name):
                    # class_name = line.strip()
                    classes_dict1[line] = 0

            print(file_name)
            txts_dir = os.path.join(self.label_dir, file_name,'labels')


            txts = os.listdir(txts_dir)

            for i_txt in tqdm(range(len(txts))):
                txt = txts[i_txt]
                # print(img)
                # img = '-nfs-安全帽反光服检测-安全帽-开源-安全帽-开源-ec74035b-1ecb-4662-8fdb-2c99ec5e31df.jpg'
                if txt[-3:] == 'txt':
                    txt_path = os.path.join(txts_dir, txt)

                    lines = open(txt_path, 'r', encoding='utf-8').readlines()
                    for line in lines:
                        label = line.strip().split(' ')[0]
    #                     if label == '1':
    #                         print(txt_path)
    #                     print(line)
                        classes_dict1 = self.sum_label(class_name[int(label)], classes_dict1)
            for key,value in classes_dict1.items():  
#                 print(key)
                print(value)
                
#                 print( '{} : {}'.format(key,value) )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_names', default=['train','valtest'], help='')
#     parser.add_argument('--file_names', default=['891'], help='')
    parser.add_argument('--label_dir',
                        default='/home/jovyan/data-vol-1/wqg/container/20220110_rec/',
#                         default='/home/jovyan/data-vol-1/wqg/container/container_rec/',
                        # /save_txt',
                        help='标签文件所在文件夹')

    return parser.parse_args()

def main(args):
    G = ProcessTxt(args)
    G.process()


if __name__ == '__main__':
    main(parse_args())