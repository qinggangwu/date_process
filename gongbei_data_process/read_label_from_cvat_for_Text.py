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
        self.height   = 0.0
        self.width    = 0.0
        self.label    = ''
        self.points   = []
        self.flag_over = bool


if __name__=='__main__':
    part_all =['1835', '1828', '1825', '1824', '1822', '1821', '1787', '1786', '1785']    # 1826号数据未标记.
    part_all =['1826']
    for part_ in part_all:
        pathdir = '/home/wqg/data/gongbei/oringe/'
        xmlpath = pathdir +part_ + '/annotations.xml'
        save_txtpath = pathdir +part_+'/labels'
        orgine_imgpath = pathdir +part_+'/imageset'
        save_imgpath = pathdir +part_+'/images'
        not_care_lable = '###'
        if not os.path.exists(save_txtpath):
            os.makedirs(save_txtpath)
        if not os.path.exists(save_imgpath):
            os.makedirs(save_imgpath)
        # 使用minidom解析器打开 XML 文档
        DOMTree = xml.dom.minidom.parse(xmlpath)
        collection = DOMTree.documentElement

        # 在集合中获取所有图片的信息
        imgmessages = collection.getElementsByTagName("image")

        # 解析每一张图片
        oneImg = datastru()
        for messages_i in tqdm(range(len(imgmessages))):
            messages = imgmessages[messages_i]

            if messages.hasAttribute("name"):
                # print("name: %s" % messages.getAttribute("name"))
                oneImg.img_path = messages.getAttribute("name")
#                 print(oneImg.img_path)
            pathTXT = os.path.basename(oneImg.img_path)
            pathTXT_ = os.path.splitext(pathTXT)[0]

            if messages.hasAttribute("width"):
                # print("width: %s" % messages.getAttribute("width"))
                oneImg.width = int(messages.getAttribute("width"))
            if messages.hasAttribute("height"):
                # print("name: %s" % messages.getAttribute("height"))
                oneImg.height = int(messages.getAttribute("height"))

            img = cv2.imread(os.path.join( orgine_imgpath,oneImg.img_path))


            # type = messages.getElementsByTagName('polygon')
            type = messages.getElementsByTagName('box')
            str_label = ''
            for i in range(len(type)):
                # label_point = {}

                value = type[i].getAttribute("points").split(';')
                xbr = int(type[i].getAttribute("xbr"))
                xtl = int(type[i].getAttribute("xtl"))
                ybr = int(type[i].getAttribute("ybr"))
                ytl = int(type[i].getAttribute("ytl"))
                valueintlist = []

                width = xbr - xtl
                height = ybr - ytl
                center_x = xtl + int(0.5 * width)
                center_y = ytl + int(0.5 * height)

                sing_label = "{} {} {} {}\n".format(round(center_x/oneImg.width,6),round(center_y/oneImg.height,6),
                                                    round(width/oneImg.width,6),round(height/oneImg.height,6))

                label_read = type[i].getAttribute("label")
                if label_read == '车辆(拱)':#'textbox':
                    label = "0"
                    str_label += label+' '+sing_label

                elif '忽略' in label_read:
                    # mask_pts = np.array([[int(xtl), int(ytl)],[int(xbr), int(ytl)],[int(xbr), int(ybr)],[int(xtl), int(ybr)]])
                    mask_pts = np.array([[xtl, ytl],[xbr, ytl],[xbr, ybr],[xtl, ybr]])
                    cv2.fillPoly(img, [mask_pts], (114, 114, 114))

                else:
                    print('label error!', label_read)


            # cv2.imshow('mask', cv2.resize(img, dsize=None, fx=0.3, fy=0.3))
            # cv2.waitKey(200000)
            if len(str_label) >0:
                txt_file = os.path.join(save_txtpath, pathTXT_ + '.txt')
                img_file = os.path.join(save_imgpath, pathTXT_ + '.jpg')
                cv2.imwrite(img_file,img)

                with open(txt_file,"w",encoding='utf-8') as f:
                    f.write(str_label)

    print('end')

