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
from math import atan2
import math

class datastru:
    def __init__(self):
        self.img_path = ''
        self.img_name = ''
        self.height   = 0.0
        self.width    = 0.0
        self.label    = ''
        self.points   = []
        self.flag_over = bool

def get_ther(value,label_read ='',img=None ,pathTXT_ = ""):
    p = []
    for i in range(1,8,2):
        point = [int(ii) for ii in value[i].split(',')]
        p.append(point)

    if label_read =='车尾':
        # maxpoint = [int((p[0][0] + p[1][0])/2), int((p[0][1] + p[1][1])/2)]
        # minpoint = [int((p[2][0] + p[3][0])/2), int((p[2][1] + p[3][1])/2)]
        maxpoint = [(p[0][0] + p[1][0])/2, (p[0][1] + p[1][1])/2]
        minpoint = [(p[2][0] + p[3][0])/2, (p[2][1] + p[3][1])/2]

    elif label_read =='车脸':
        # minpoint = [int((p[0][0] + p[1][0]) / 2), int((p[0][1] + p[1][1]) / 2)]
        # maxpoint = [int((p[2][0] + p[3][0]) / 2), int((p[2][1] + p[3][1]) / 2)]
        minpoint = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2]
        maxpoint = [(p[2][0] + p[3][0]) / 2, (p[2][1] + p[3][1]) / 2]
    else:
        return 181
        # return 0,0,0,0,0,0,0,0

    x ,y = maxpoint[0]- minpoint[0] ,maxpoint[1] -minpoint[1]
    ther = math.atan2(x,y)

    # print(x ,y ,ther/math.pi*180)
    if label_read == '车尾' and ( ther/math.pi*180 + 180 >270 or ther/math.pi*180 + 180 < 90 ):
        return 181
    elif label_read == '车脸' and   90 < ther/math.pi*180 + 180 < 270:
        return 181


    return (ther / math.pi * 180 + 180) //2


if __name__=='__main__':
    part_all =['1835', '1828', '1825', '1824', '1822', '1821', '1787', '1786', '1785']    # 1826号数据未标记.
    part_all =['1964']
    labeldict = {'行人':0,'车脸轿车':1,'车脸卡车':2,'车脸大巴':3,'车脸特殊车辆':4,'车尾轿车':5,'车尾卡车':6,'车尾大巴':7,'车尾特殊车辆':8}
    for part_ in part_all:
        pathdir = '/home/wqg/data/maxvision_data/ADAS/'
        xmlpath = os.path.join(pathdir , part_, 'annotations.xml')
        orgine_imgpath = os.path.join(pathdir,  part_, 'imageset')

        savetrain_txtpath = os.path.join(pathdir,part_, 'train', 'labels')
        savetrain_imgpath = os.path.join(pathdir,part_, 'train', 'images')
        saveval_txtpath = os.path.join(pathdir,  part_,'valtest', 'labels')
        saveval_imgpath = os.path.join(pathdir, part_, 'valtest', 'images')
        not_care_lable = '###'

        os.makedirs(savetrain_txtpath,exist_ok=True)
        os.makedirs(savetrain_imgpath,exist_ok=True)
        os.makedirs(saveval_txtpath,exist_ok=True)
        os.makedirs(saveval_imgpath,exist_ok=True)

        # 使用minidom解析器打开 XML 文档
        # xmlpath = '/home/wqg/data/公司数据/annotations.xml'
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
            pathTXT_ = os.path.splitext(pathTXT)[0][-4:]

            if messages.hasAttribute("width"):
                # print("width: %s" % messages.getAttribute("width"))
                oneImg.width = int(messages.getAttribute("width"))
            if messages.hasAttribute("height"):
                # print("name: %s" % messages.getAttribute("height"))
                oneImg.height = int(messages.getAttribute("height"))

            img = cv2.imread(os.path.join(orgine_imgpath,oneImg.img_name))

            # 获取标记框
            type = messages.getElementsByTagName('point')
            if type == []:
                continue
            str_label = ''
            for i in range(len(type)):
                value = type[i].getAttribute("points").split(';')

                # print('value-type',value)
                X = [int(value[i].split(',')[0]) for i in range(4)]
                Y = [int(value[i].split(',')[1]) for i in range(4)]
                # print('xy',X,Y)


                xbr = max(X) if max(X) < oneImg.width else oneImg.width
                xtl = min(X) if min(X) > 0 else 0
                ybr = max(Y) if max(Y) < oneImg.height else oneImg.height
                ytl = min(Y) if min(Y) > 0 else 0


                valueintlist = []
                width = xbr - xtl
                height = ybr - ytl
                center_x = xtl + int(0.5 * width)
                center_y = ytl + int(0.5 * height)

                if center_x<0 or center_y<0 or center_x>oneImg.width or center_y>oneImg.height:
                    print('center_x,center_y',center_x,center_y)
                    print(xbr, xtl, ybr, ytl)
                # if round(center_x/oneImg.width,6) <0:
                #     print(center_x,oneImg.width,)
                #     print(xbr, xtl, ybr, ytl)
                #     print(round(center_x/oneImg.width,6))

                label_read = type[i].getAttribute("label")

                if '人' in label_read:
                    labelD = label_read
                    points=[-1,-1,-1,-1,-1,-1,-1,-1]
                else:
                    name_read = type[i].getElementsByTagName('attribute')[0].firstChild.data
                    labelD = label_read+name_read
                    # points =[[int(ii) for ii in value[i].split(',')]for i in range(1, 8, 2)]
                    points =[int(ii) for i in range(1, 8, 2) for ii in value[i].split(',')]

                ther = get_ther(value,label_read, img ,pathTXT_)

                sing_label = "{} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                   round(center_x / oneImg.width, 6),
                   round(center_y / oneImg.height, 6),
                   round(width / oneImg.width, 6),
                   round(height / oneImg.height, 6),
                   round(points[0] / oneImg.width, 6),
                   round(points[1] / oneImg.height, 6),
                   round(points[2] / oneImg.width, 6),
                   round(points[3] / oneImg.height, 6),
                   round(points[4] / oneImg.width, 6),
                   round(points[5] / oneImg.height, 6),
                   round(points[6] / oneImg.width, 6),
                   round(points[7] / oneImg.height, 6)
                   )

                label = str(labeldict[labelD])   # "0"
                str_label += label+' '+sing_label +' '
                # print(labelD , label)


            # 获取忽略框
            box = messages.getElementsByTagName('box')
            # print('box',box)
            # print(box[0].getAttribute("points"));quit()
            # str_label = ''
            for i in range(len(box)):
                xbr = int(box[i].getAttribute("xbr"))
                xtl = int(box[i].getAttribute("xtl"))
                ybr = int(box[i].getAttribute("ybr"))
                ytl = int(box[i].getAttribute("ytl"))
                # print(ind,'xxyy', xbr, xtl, ybr, ytl)
                mask_pts = np.array([[xtl, ytl], [xbr, ytl], [xbr, ybr], [xtl, ybr]])
                # print(mask_pts)
                cv2.fillPoly(img, [mask_pts], (114, 114, 114))


            # cv2.imshow('mask', cv2.resize(img, dsize=None, fx=0.3, fy=0.3))
            # cv2.waitKey(200000)
            if len(str_label) >0:
                if ind %10 ==0:  # 20% 作为验证集
                    txt_file = os.path.join(saveval_txtpath, pathTXT_ + '.txt')
                    img_file = os.path.join(saveval_imgpath, pathTXT_ + '.jpg')

                    # shutil.copy(os.path.join(orgine_imgpath,oneImg.img_name) ,img_file)

                    cv2.imwrite(img_file,img)

                    with open(txt_file,"w",encoding='utf-8') as f:
                        f.write(str_label)
                else:
                    txt_file = os.path.join(savetrain_txtpath, pathTXT_ + '.txt')
                    img_file = os.path.join(savetrain_imgpath, pathTXT_ + '.jpg')
                    # shutil.copy(os.path.join(orgine_imgpath, oneImg.img_name), img_file)
                    cv2.imwrite(img_file, img)

                    with open(txt_file, "w", encoding='utf-8') as f:
                        f.write(str_label)

    print('end')

