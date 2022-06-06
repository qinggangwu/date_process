



# -*- coding: utf-8 -*-
import random
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
import os

from tqdm import tqdm
import shutil
import cv2
import numpy as np


provincelist = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]

# --- 绘制边界框

def DrawBox(im, box):
    draw = ImageDraw.Draw(im)
    draw.rectangle([tuple(box[0]), tuple(box[1])],  outline="#FFFFFF", width=3)

# --- 绘制四个关键点

def DrawPoint(im, points):

    draw = ImageDraw.Draw(im)

    for p in points:
        center = (p[0], p[1])
        radius = 5
        right = (center[0]+radius, center[1]+radius)
        left = (center[0]-radius, center[1]-radius)
        draw.ellipse((left, right), fill="#FF0000")

# --- 绘制车牌

def DrawLabel(im, label):
    draw = ImageDraw.Draw(im)
   # draw.multiline_text((30,30), label.encode("utf-8"), fill="#FFFFFF")
    font = ImageFont.truetype('simsun.ttc', 64)
    draw.text((30, 30), label, font=font)

# --- 图片可视化

def ImgShow(imgpath, box, points, label):
    # 打开图片
    im = Image.open(imgpath)
    DrawBox(im, box)
    # DrawPoint(im, points)
    # DrawLabel(im, label)
    # 显示图片
    im.show()
    # im.save('result.jpg')


def analyse_CCPD():
    # 图像路径
    dirPath ='/home/wqg/data/CCPD2020/ccpd_green/'

    imgpath = dirPath + 'val/0136360677083-95_103-255&434_432&512-432&512_267&494_255&434_424&449-0_0_3_25_30_24_24_32-98-218.jpg'

    # 图像名
    imgname = os.path.basename(imgpath).split('.')[0]

    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = imgname.split('-')

    # --- 边界框信息
    box = box.split('_')
    box = [list(map(int, i.split('&'))) for i in box]
    # print(box)
    # pad = 20
    # box[0] = [box[0][0] - int(pad /4) , box[0][1] - pad]
    # box[1] = [box[1][0] + int(pad /4) , box[1][1] + pad]
    # print(box)

    # --- 关键点信息
    points = points.split('_')
    points = [list(map(int, i.split('&'))) for i in points]
    # 将关键点的顺序变为从左上顺时针开始
    points = points[-2:]+points[:2]

    # --- 读取车牌号
    label = label.split('_')
    # 省份缩写
    province = provincelist[int(label[0])]
    # 车牌信息
    words = [wordlist[int(i)] for i in label[1:]]
    # 车牌号
    label = province+''.join(words)

    # --- 图片可视化
    ImgShow(imgpath, box, points, label)



class make_plate_imgs:

    def __init__(self,root):
        self.root = root
        self.filelist = [ '蓝牌' ,'黄牌', '双层黄牌', '绿牌' , '黄绿牌','白牌', '双层白牌', '黑牌', '双层绿牌']
        self.namedict = {'蓝牌': 0, '黄牌': 1, '双层黄牌': 2, '绿牌': 3, '黄绿牌': 4, '白牌': 5, '双层白牌': 6, '黑牌': 7, '双层绿牌': 8, '港澳车牌': 7}


    def get_bgpath_list(self):

        with open(os.path.join(self.root,'CCPD2019/splits/val.txt') ,'r' ) as f:
            readinfo = f.readlines()

        return [os.path.join(self.root,'CCPD2019', ii[:-1]) for ii in readinfo]

    def get_img_list(self,path):
        xmlpath = os.path.join(path, 'annotations.xml')
        info = []

        # xmlpath = os.path.join(data_root, part, 'annotations.xml')
        tree = ET.ElementTree(file=xmlpath)
        root = tree.getroot()
        child_root = root[1:]

        for idx, child_of_image in enumerate(child_root):
            imginfo = child_of_image.attrib
            # filename = "%06d"%idx
            size = (int(imginfo['width']) ,  int(imginfo['height']))      # ( annotation.imgWidth , annotation.imgHeight )

            xy = []
            for child_of_point in child_of_image:
                points = child_of_point.attrib['points'].split(";")
                polygon =[tuple(map(int, ii.split(',') )) for ii in points]
                xy.append(polygon[0])
                xy.append(polygon[2])

            X = [ ii[0] for ii in xy]
            Y = [ ii[1] for ii in xy]


            pad = 4
            # 坐标超出图片边界,按照边界进行切分
            xbr = max(X)+pad if max(X)+pad < size[0] else size[0]
            xtl = min(X)-pad if min(X)-pad  > 0 else 0
            ybr = max(Y)+pad if max(Y)+pad < size[1] else size[1]
            ytl = min(Y)-pad if min(Y)-pad > 0 else 0

            info.append((imginfo, ([xtl, ytl] ,[xbr,ybr])) )


        # info  = [ (ii.attrib , ii.attrib['points'])  for ii in child_root]

        return info


    def make_tansforme(self,bgpath , imgpath ,imginfolist):

        # 整理背景图片坐标信息
        imgname = os.path.basename(bgpath).split('.')[0]
        _, _, box, points, label, brightness, blurriness = imgname.split('-')

        # --- 关键点信息
        points = points.split('_')
        points = [list(map(int, i.split('&'))) for i in points]
        # 将关键点的顺序变为从左上顺时针开始
        bgpoints = points[-2:] + points[:2]

        X = [ii[0] for ii in bgpoints]
        Y = [ii[1] for ii in bgpoints]


        # 整理 imginfolist
        imgdict = imginfolist[0]
        point1 = imginfolist[1][0]
        point2 = imginfolist[1][1]

        # 读取 背景图片和 前景替换图片
        bg = cv2.imread(bgpath)
        img = cv2.imread(imgpath)


        # 切取 替换图片
        img = img[point1[1]:point2[1], point1[0]:point2[0]]

        # 得到 bgimg 透视变换矩阵  进行透视变换
        height, width = img.shape[:2]
        bgheight, bgwidth = bg.shape[:2]

        mat_src = np.float32(bgpoints)
        mat_dst = np.float32([bgpoints[0], [bgpoints[0][0] + width, bgpoints[0][1]],
                              [bgpoints[0][0] + width, bgpoints[0][1] + height], [bgpoints[0][0], bgpoints[0][1] + height]])


        Minv = cv2.getPerspectiveTransform(mat_dst, mat_src)

        makebg = np.ones((bgheight, bgwidth,4)).astype('uint8') * 255
        make_img_a = np.zeros((height,width)).astype('uint8')
        # MM = cv2.getPerspectiveTransform(mat_src, mat_dst)
        # bg = cv2.warpPerspective(makebg, MM, (bgwidth*2, bgheight))

        makebg[bgpoints[0][1] : bgpoints[0][1] +height ,bgpoints[0][0] : bgpoints[0][0]+width ] = cv2.merge((img, make_img_a))

        dst = cv2.warpPerspective(makebg, Minv, (bgwidth, bgheight))

        # 对bg图片进行裁剪 通过alpth通道进行原图修改
        # --- 边界框信息
        # boxx = box.split('_')
        # boxx = [list(map(int, i.split('&'))) for i in boxx]

        box = [[min(X),min(Y)], [max(X),max(Y)]]
        padx = int(box[0][0]/5)
        pady = int(box[0][1]/8)
        # print(boxx ,box)
        box1 = [box[0][0]- padx , box[0][1]-pady ]
        box2 = [box[1][0]+ padx , box[1][1]+pady ]


        crop_plate = dst[box1[1] : box2[1] ,box1[0] : box2[0] ]

        al = crop_plate[:,:,3]/255

        for i in range(3):
            bg[box1[1]:box2[1], box1[0]:box2[0], i] = bg[box1[1]:box2[1], box1[0]:box2[0], i] * (al) + crop_plate[:,:,i] * (1 - al)

        return bg,box


    def make_plate_img(self):

        num_dict ={'黄牌': 10000 , '双层黄牌': 8000,  '黄绿牌': 5000, '白牌': 8000 , '双层白牌': 5000, '黑牌': 5000, '双层绿牌': 5000, '港澳车牌': 5000}
        num_dict ={'港澳车牌': 5000}
        bglist = self.get_bgpath_list()

        savedir = "/home/wqg/data/maxtest/"
        savetrain_txtpath = os.path.join(savedir, 'det', 'train', 'labels')
        savetrain_imgpath = os.path.join(savedir, 'det', 'train', 'images')
        saveval_txtpath = os.path.join(savedir, 'det', 'valtest', 'labels')
        saveval_imgpath = os.path.join(savedir, 'det', 'valtest', 'images')

        os.makedirs(savetrain_txtpath, exist_ok=True)
        os.makedirs(savetrain_imgpath, exist_ok=True)
        os.makedirs(saveval_imgpath, exist_ok=True)
        os.makedirs(saveval_imgpath, exist_ok=True)


        for key in num_dict:

            # imginfolist = self.get_img_list(os.path.join(self.root, key))
            imginfolist = self.get_img_list('/home/wqg/data/maxvision_data/car_plate/' + key)

            # for  i in num_dict[key]:
            for i in tqdm(range(1000)):
                bgpath = random.choice(bglist)

                imginfo = random.choice(imginfolist)
                imgpath = os.path.join(self.root, key,'imageset',imginfo[0]['name'])
                imgpath = os.path.join(self.root, 'maxvision_data/car_plate', key,'imageset',imginfo[0]['name'])

                resultimg ,box = self.make_tansforme(bgpath,imgpath,imginfo)

                size = resultimg.shape

                cv2.rectangle(resultimg,box[0],box[1],(0,0,255),1)

                width = box[1][0] - box[0][0]
                height = box[1][1] - box[0][1]
                center_x = box[0][0] + int(0.5 * width)
                center_y = box[0][1] + int(0.5 * height)

                sing_label = "{} {} {} {} {}\n".format(self.namedict[key],
                                                       round(center_x / size[1], 6),
                                                       round(center_y / size[0], 6),
                                                       round(width / size[1], 6),
                                                       round(height / size[0], 6))


                # print(sing_label)

                if i % 10 == 0:
                    new_imgpath = os.path.join(saveval_imgpath, os.path.basename(bgpath))
                    new_labelpath = os.path.join(saveval_txtpath, os.path.basename(bgpath)[:-4] + 'txt')

                    # shutil.copy(or_img_path, new_imgpath)
                    cv2.imwrite(new_imgpath, resultimg)
                    with open(new_labelpath, "w", encoding='utf-8') as f:
                        f.write(sing_label)

                else:
                    new_imgpath = os.path.join(savetrain_imgpath, os.path.basename(bgpath))
                    new_labelpath = os.path.join(savetrain_txtpath, os.path.basename(bgpath)[:-4] + 'txt')

                    cv2.imwrite(new_imgpath, resultimg)
                    with open(new_labelpath, "w", encoding='utf-8') as f:
                        f.write(sing_label)


                # cv2.imwrite('/home/wqg/data/maxvision_data/car_plate/test/'+os.path.basename(bgpath) , resultimg)
                # new_labelpath = '/home/wqg/data/maxvision_data/car_plate/test/'+os.path.basename(bgpath)[:-3] +'txt'
                # with open(new_labelpath, "w", encoding='utf-8') as f:
                #         f.write(sing_label)



def precess_CCPD_grenn():
    root = '/home/jovyan/data-vol-1/wqg/car_plate/'
    savedir = "/home/jovyan/data-vol-1/wqg/car_make_plate_det"

    savetrain_txtpath = os.path.join(savedir, 'train', 'labels')
    savetrain_imgpath = os.path.join(savedir, 'train', 'images')
    saveval_txtpath = os.path.join(savedir, 'valtest', 'labels')
    saveval_imgpath = os.path.join(savedir, 'valtest', 'images')

    os.makedirs(savetrain_txtpath, exist_ok=True)
    os.makedirs(savetrain_imgpath, exist_ok=True)
    os.makedirs(saveval_txtpath, exist_ok=True)
    os.makedirs(saveval_imgpath, exist_ok=True)

    filelist = os.listdir(os.path.join(root,'CCPD2020/ccpd_green'))
    for file in filelist:
        for idx,imgname in enumerate(os.listdir(os.path.join(root,'CCPD2020/ccpd_green', file))):
            _, _, box, points, label, brightness, blurriness = imgname.split('-')
            points = points.split('_')
            points = [list(map(int, i.split('&'))) for i in points]
            # 将关键点的顺序变为从左上顺时针开始
            points = points[-2:] + points[:2]

            X = [ii[0] for ii in points]
            Y = [ii[1] for ii in points]

            box = [[min(X), min(Y)], [max(X), max(Y)]]

            width = box[1][0] - box[0][0]
            height = box[1][1] - box[0][1]
            center_x = box[0][0] + int(0.5 * width)
            center_y = box[0][1] + int(0.5 * height)

            imgpath = os.path.join(root, 'CCPD2020/ccpd_green', file, imgname)

            img = cv2.imread(imgpath)

            size = img.shape
            sing_label = "{} {} {} {} {}\n".format(3,
                                                   round(center_x / size[1], 6),
                                                   round(center_y / size[0], 6),
                                                   round(width / size[1], 6),
                                                   round(height / size[0], 6))

            if idx % 10 == 0:
                new_imgpath = os.path.join(saveval_imgpath, imgname)
                new_labelpath = os.path.join(saveval_txtpath, imgname[:-3] + 'txt')

                shutil.copy(imgpath, new_imgpath)
                # cv2.imwrite(new_imgpath, resultimg)
                with open(new_labelpath, "w", encoding='utf-8') as f:
                    f.write(sing_label)

            else:
                new_imgpath = os.path.join(savetrain_imgpath, imgname)
                new_labelpath = os.path.join(savetrain_txtpath, imgname[:-3] + 'txt')

                shutil.copy(imgpath, new_imgpath)
                # cv2.imwrite(new_imgpath, resultimg)
                with open(new_labelpath, "w", encoding='utf-8') as f:
                    f.write(sing_label)


            # cv2.rectangle(img,box[0],box[1],(0,0,255),1)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)

def precess_CCPD_blue():
    root = '/home/jovyan/data-vol-1/wqg/car_plate/'
    savedir = "/home/jovyan/data-vol-1/wqg/car_make_plate_det"

    savetrain_txtpath = os.path.join(savedir, 'train', 'labels')
    savetrain_imgpath = os.path.join(savedir, 'train', 'images')
    saveval_txtpath = os.path.join(savedir, 'valtest', 'labels')
    saveval_imgpath = os.path.join(savedir, 'valtest', 'images')

    os.makedirs(savetrain_txtpath, exist_ok=True)
    os.makedirs(savetrain_imgpath, exist_ok=True)
    os.makedirs(saveval_txtpath, exist_ok=True)
    os.makedirs(saveval_imgpath, exist_ok=True)


    with open(os.path.join(root,'CCPD2019/splits/make_blue.txt'),'r',encoding='utf-8') as f:
        imgnamelist = f.readlines()

    imglist = [ os.path.join(root , 'CCPD2019' , name)  for name in imgnamelist]

    imglist = list(set(random.choices(imglist,k=12000)))

    for idx,name in enumerate(imglist):
        if not os.path.exists(name):
            continue

        imgname = os.path.basename(name).split('.')[0]
        _, _, box, points, label, brightness, blurriness = imgname.split('-')
        points = points.split('_')
        points = [list(map(int, i.split('&'))) for i in points]
        # 将关键点的顺序变为从左上顺时针开始
        points = points[-2:] + points[:2]

        X = [ii[0] for ii in points]
        Y = [ii[1] for ii in points]

        box = [[min(X), min(Y)], [max(X), max(Y)]]

        width = box[1][0] - box[0][0]
        height = box[1][1] - box[0][1]
        center_x = box[0][0] + int(0.5 * width)
        center_y = box[0][1] + int(0.5 * height)

        img = cv2.imread(name)

        size = img.shape
        sing_label = "{} {} {} {} {}\n".format(3,
                                               round(center_x / size[1], 6),
                                               round(center_y / size[0], 6),
                                               round(width / size[1], 6),
                                               round(height / size[0], 6))

        if idx % 10 == 0:
            new_imgpath = os.path.join(saveval_imgpath, imgname)
            new_labelpath = os.path.join(saveval_txtpath, imgname[:-3] + 'txt')

            shutil.copy(name, new_imgpath)
            # cv2.imwrite(new_imgpath, resultimg)
            with open(new_labelpath, "w", encoding='utf-8') as f:
                f.write(sing_label)

        else:
            new_imgpath = os.path.join(savetrain_imgpath, imgname)
            new_labelpath = os.path.join(savetrain_txtpath, imgname[:-3] + 'txt')

            shutil.copy(name, new_imgpath)
            # cv2.imwrite(new_imgpath, resultimg)
            with open(new_labelpath, "w", encoding='utf-8') as f:
                f.write(sing_label)


            # cv2.rectangle(img,box[0],box[1],(0,0,255),1)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)


def main():
    root = '/home/wqg/data'

    Precess = make_plate_imgs(root)
    Precess.make_plate_img()




if __name__ =="__main__":

    # main()
    precess_CCPD_grenn()
    precess_CCPD_blue

    # print(os.path.exists('/home/wqg/data/maxvision_data/car_plate/test/0125-91_86-235&509_422&580-421&580_237&578_238&513_422&515-0_0_18_8_27_25_24-102-31.jpg'))
    # print(os.path.exists('/home/wqg/data/maxvision_data/car_plate/test/0125-91_86-235&509_422&580-421&580_237&578_238&513_422&515-0_0_18_8_27_25_24-102-31.jpg'))
