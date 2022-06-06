"""
集装箱的字符识别数据的标签处理：将数据标签从cvat中读取出来，按照一张图片一个txt文件进行保存
"""

from xml.dom.minidom import parse
import xml.dom.minidom
import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np


class datastru:
    def __init__(self):
        self.img_path = ''
        self.img_name = ''
        self.height   = 0.0
        self.width    = 0.0
        self.label    = ''
        self.points   = []
        self.flag_over = bool


if __name__=='__main__':
    part_all =['1835', '1828', '1825', '1824', '1822', '1821', '1787', '1786', '1785']    # 1826号数据未标记.
    part_all =['1826']
    labeldict = {'行人':0,'车脸轿车':1,'车脸卡车':2,'车脸大巴':3,'车脸特殊车辆':4,'车尾轿车':5,'车尾卡车':6,'车尾大巴':7,'车尾特殊车辆':8,}
    # labeldict_num = {'行人':0,'骑车行人':0,'车脸轿车':0,'车脸卡车':0,'车脸大巴':0,'车脸特殊车辆':0,'车尾轿车':0,'车尾卡车':0,'车尾大巴':0,'车尾特殊车辆':0,}
    labeldict_num = {'行人':0,'骑车行人':0,'三轮车载人':0,'轿车':0,'卡车':0,'大巴':0,'特殊车辆':0,'巴士':0,'货车':0,'环卫车':0,'油罐车':0,'箱式货车':0,'皮卡':0,'平板拖车':0,'异型车':0,'工程车':0}

    for part_ in part_all:
        pathdir = '/home/data-vol-1/wqg/ADAS/'
        xmlpath = os.path.join(pathdir ,'oringe', part_, 'annotations.xml')
        orgine_imgpath = os.path.join(pathdir, 'oringe', part_, 'imageset')

        # savetrain_txtpath = os.path.join(pathdir,'det', 'train', 'labels')
        # savetrain_imgpath = os.path.join(pathdir, 'det','train', 'images')
        # saveval_txtpath = os.path.join(pathdir, 'det', 'valtest', 'labels')
        # saveval_imgpath = os.path.join(pathdir, 'det', 'valtest', 'images')
        not_care_lable = '###'

        # os.makedirs(savetrain_txtpath,exist_ok=True)
        # os.makedirs(savetrain_imgpath,exist_ok=True)
        # os.makedirs(saveval_imgpath,exist_ok=True)
        # os.makedirs(saveval_imgpath,exist_ok=True)

        # 使用minidom解析器打开 XML 文档
        xmlpath = '/home/wqg/data/maxvision_data/annotationsky.xml'
        DOMTree = xml.dom.minidom.parse(xmlpath)
        collection = DOMTree.documentElement

        # 在集合中获取所有图片的信息
        imgmessages = collection.getElementsByTagName("image")

        # 解析每一张图片
        oneImg = datastru()
        for ind in tqdm(range(len(imgmessages))):
        # for ind in tqdm(range(500)):
            messages = imgmessages[ind]

            if messages.hasAttribute("name"):
                # print("name: %s" % messages.getAttribute("name"))
                oneImg.img_name = messages.getAttribute("name")
#                 print(oneImg.img_path)
            pathTXT = os.path.basename(oneImg.img_name)
            pathTXT_ = os.path.splitext(pathTXT)[0]

            if messages.hasAttribute("width"):
                # print("width: %s" % messages.getAttribute("width"))
                oneImg.width = int(messages.getAttribute("width"))
            if messages.hasAttribute("height"):
                # print("name: %s" % messages.getAttribute("height"))
                oneImg.height = int(messages.getAttribute("height"))

            # img = cv2.imread(os.path.join(orgine_imgpath,oneImg.img_name))

            # 获取标记框
            type = messages.getElementsByTagName('point')
            if type == []:
                continue
            str_label = ''
            for i in range(len(type)):
                # value = type[i].getAttribute("points").split(';')
                #
                # X = [int(value[i].split(',')[0]) for i in range(4)]
                # Y = [int(value[i].split(',')[1]) for i in range(4)]
                # # print('xy',X,Y)
                #
                #
                # xbr = max(X) if max(X) < oneImg.width else oneImg.width
                # xtl = min(X) if min(X) > 0 else 0
                # ybr = max(Y) if max(Y) < oneImg.height else oneImg.height
                # ytl = min(Y) if min(Y) > 0 else 0
                #
                #
                # valueintlist = []
                # width = xbr - xtl
                # height = ybr - ytl
                # center_x = xtl + int(0.5 * width)
                # center_y = ytl + int(0.5 * height)
                #
                # if center_x<0 or center_y<0 or center_x>oneImg.width or center_y>oneImg.height:
                #     print('center_x,center_y',center_x,center_y)
                #     print(xbr, xtl, ybr, ytl)
                # # if round(center_x/oneImg.width,6) <0:
                # #     print(center_x,oneImg.width,)
                # #     print(xbr, xtl, ybr, ytl)
                # #     print(round(center_x/oneImg.width,6))
                #
                # sing_label = "{} {} {} {}\n".format(round(center_x/oneImg.width,6),round(center_y/oneImg.height,6),
                #                                     round(width/oneImg.width,6),round(height/oneImg.height,6))

                label_read = type[i].getAttribute("label")

                if '人' in label_read:
                    labelD = label_read
                else:
                    # try:
                    name_read = type[i].getElementsByTagName('attribute')[0].firstChild.data
                    # except:
                    #     print(i,label_read)
                    # labelD = label_read+name_read
                    labelD = name_read
                labeldict_num[labelD] += 1


                # label = str(labeldict[labelD])   # "0"
                # str_label += label+' '+sing_label
                # # print(labelD , label)
            # # 获取忽略框
            # box = messages.getElementsByTagName('box')
            #
            # for i in range(len(box)):
            #     xbr = int(box[i].getAttribute("xbr"))
            #     xtl = int(box[i].getAttribute("xtl"))
            #     ybr = int(box[i].getAttribute("ybr"))
            #     ytl = int(box[i].getAttribute("ytl"))
            #     # print(ind,'xxyy', xbr, xtl, ybr, ytl)
            #     mask_pts = np.array([[xtl, ytl], [xbr, ytl], [xbr, ybr], [xtl, ybr]])
            #     # print(mask_pts)
            #     # cv2.fillPoly(img, [mask_pts], (114, 114, 114))

    print(labeldict_num)
    print('end')



"""
5月7日统计

1992  20220328-晴天
{'行人': 554, '骑车行人': 312, '三轮车载人': 0, '轿车': 6159, '卡车': 53, '大巴': 0, '特殊车辆': 0, '巴士': 463, '货车': 129, '环卫车': 13, '油罐车': 1, '箱式货车': 376, '皮卡': 6, '平板拖车': 0, '异型车': 0, '工程车': 8}

1993  20220329-晴天
{'行人': 226, '骑车行人': 145, '三轮车载人': 3, '轿车': 2088, '卡车': 42, '大巴': 0, '特殊车辆': 0, '巴士': 206, '货车': 56, '环卫车': 26, '油罐车': 0, '箱式货车': 27, '皮卡': 1, '平板拖车': 1, '异型车': 0, '工程车': 17}

1994  20220402-晴天
{'行人': 60, '骑车行人': 31, '三轮车载人': 5, '轿车': 486, '卡车': 10, '大巴': 0, '特殊车辆': 0, '巴士': 2, '货车': 11, '环卫车': 1, '油罐车': 1, '箱式货车': 11, '皮卡': 0, '平板拖车': 0, '异型车': 9, '工程车': 7}


2000  2001 2002  2011  2012  开源数据    5000张图片
{'行人': 1546, '骑车行人': 1301, '三轮车载人': 346, '轿车': 10242, '卡车': 716, '大巴': 0, '特殊车辆': 0, '巴士': 510, '货车': 373, '环卫车': 45, '油罐车': 7, '箱式货车': 1158, '皮卡': 42, '平板拖车': 27, '异型车': 55, '工程车': 59}


"""



"""
公开数据集
{'行人': 6455, '骑车行人': 1215, '车脸轿车': 10784, '车脸卡车': 509, '车脸大巴': 113, '车脸特殊车辆': 45, '车尾轿车': 11876, '车尾卡车': 452, '车尾大巴': 413, '车尾特殊车辆': 96}

tusimpl & culane
{'行人': 246, '骑车行人': 158, '车脸轿车': 424, '车脸卡车': 44, '车脸大巴': 15, '车脸特殊车辆': 3, '车尾轿车': 2228, '车尾卡车': 441, '车尾大巴': 31, '车尾特殊车辆': 51}
"""


""""

1983  2022.3.22-1 阴天
{'行人': 277, '骑车行人': 219, '轿车': 3265, '卡车': 14, '大巴': 0, '特殊车辆': 0, '巴士': 264, '货车': 105, '环卫车': 53, '油罐车': 0, '箱式货车': 26, '皮卡': 2, '平板拖车': 4, '异型车': 0, '工程车': 13}


1984 2022.3.22-2  阴天
{'行人': 520, '骑车行人': 184, '轿车': 4252, '卡车': 36, '大巴': 0, '特殊车辆': 0, '巴士': 490, '货车': 67, '环卫车': 51, '油罐车': 0, '箱式货车': 13, '皮卡': 1, '平板拖车': 1, '异型车': 0, '工程车': 38}

1985  2022.3.24雨天
{'行人': 350, '骑车行人': 264, '轿车': 3419, '卡车': 31, '大巴': 0, '特殊车辆': 0, '巴士': 391, '货车': 62, '环卫车': 0, '油罐车': 0, '箱式货车': 13, '皮卡': 0, '平板拖车': 0, '异型车': 0, '工程车': 1}


"""
