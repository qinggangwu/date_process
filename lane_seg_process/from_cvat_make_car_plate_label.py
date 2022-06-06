
import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil


def read_cavt():

    data_root = '/home/wqg/data/maxvision_data'
    namedict = {'蓝牌': 0 , '黄牌': 1 , '双层黄牌': 2 , '绿牌': 3 , '黄绿牌': 4 , '白牌': 5 , '双层白牌': 6 , '黑牌': 7 , '双层绿牌': 8 }

    imgdir = "/home/wqg/data/maxvision_data/segment/imageset"
    savedir = "/home/wqg/data/maxtest/"


    savetrain_txtpath = os.path.join(savedir, 'det', 'train', 'labels')
    savetrain_imgpath = os.path.join(savedir, 'det', 'train', 'images')
    saveval_txtpath = os.path.join(savedir, 'det', 'valtest', 'labels')
    saveval_imgpath = os.path.join(savedir, 'det', 'valtest', 'images')

    os.makedirs(savetrain_txtpath,exist_ok=True)
    os.makedirs(savetrain_imgpath,exist_ok=True)
    os.makedirs(saveval_txtpath,exist_ok=True)
    os.makedirs(saveval_imgpath,exist_ok=True)

    for part in os.listdir(data_root):

        xmlpath =  os.path.join(data_root, part , 'annotations.xml')
        tree = ET.ElementTree(file=xmlpath)
        root = tree.getroot()
        child_root = root[1:]


        label_num = namedict[part]
        # label_num = random.randint(0,4)

        for idx, child_of_image in tqdm(enumerate(child_root)):
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

            # 坐标超出图片边界,按照边界进行切分
            xbr = max(X)+2 if max(X)+2 < size[0] else size[0]
            xtl = min(X)-2 if min(X)-2  > 0 else 0
            ybr = max(Y)+2 if max(Y)+2 < size[1] else size[1]
            ytl = min(Y)-2 if min(Y)-2 > 0 else 0

            valueintlist = []
            width = xbr - xtl
            height = ybr - ytl
            center_x = xtl + int(0.5 * width)
            center_y = ytl + int(0.5 * height)

            if center_x < 0 or center_y < 0 or center_x > size[0] or center_y > size[1]:
                print('center_x,center_y', center_x, center_y)
                print(xbr, xtl, ybr, ytl)

            sing_label = "{} {} {} {} {}\n".format(label_num ,round(center_x / size[0], 6), round(center_y /size[1], 6),
                                                round(width /size[0], 6), round(height / size[1], 6))

            or_img_path = os.path.join(data_root, part, 'imageset', imginfo["name"])

            if idx%10 == 0 :
                new_imgpath = os.path.join(saveval_imgpath , imginfo["name"])
                new_labelpath = os.path.join(saveval_txtpath , imginfo["name"][:-3] + 'txt')

                shutil.copy(or_img_path , new_imgpath)
                with open(new_labelpath, "w", encoding='utf-8') as f:
                        f.write(sing_label)

            else:
                new_imgpath = os.path.join(savetrain_imgpath, imginfo["name"])
                new_labelpath = os.path.join(savetrain_txtpath, imginfo["name"][:-3] + 'txt')

                shutil.copy(or_img_path, new_imgpath)
                with open(new_labelpath, "w", encoding='utf-8') as f:
                    f.write(sing_label)


if __name__ == "__main__":
    read_cavt()








