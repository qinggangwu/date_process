"""
车道线数据处理:将数据读取出来,整理成mask
"""

from xml.dom.minidom import parse
import xml.dom.minidom
import os
import shutil
from tqdm import tqdm
import numpy as np
import cv2


class datastru:
    def __init__(self):
        self.img_path = ''
        self.height   = 0.0
        self.width    = 0.0
        self.label    = ''
        self.points   = []
        self.flag_over = bool

def draw(im, line, idx, shape, show=False):
    '''
    Generate the segmentation label according to json annotation
    '''
    # line_x = line[::2]
    # line_y = line[1::2]
    line_x = [int(ii.split(",")[0]) for ii in line]
    line_y = [int(ii.split(",")[1]) for ii in line]

    pt0 = (int((line_x[0])/shape[1] *640), int((line_y[0]-24)/shape[0] *360))
    # pt0 = (int(line_x[0]), int(line_y[0]))

    if show:
        cv2.putText(im, str(idx), (int(line_x[len(line_x) // 2]), int(line_y[len(line_x) // 2]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = idx * 60

    for i in range(len(line_x) - 1):
        cv2.line(im, pt0, (int(line_x[i + 1]/shape[1] *640), int((line_y[i + 1]-24)/shape[0] *360)), (idx*60,), thickness=2)  # thickness = i//4 +1
        pt0 = (int(line_x[i + 1]/shape[1] *640), int((line_y[i + 1]-24)/shape[0] *360))
        # cv2.line(im, pt0, (int(line_x[i + 1]), int(line_y[i + 1] )), (idx * 60,), thickness=2)  # thickness = i//4 +1
        # pt0 = (int(line_x[i + 1] ), int(line_y[i + 1] ))
    # cv2.imshow('im', im)
    # cv2.waitKey(1000)


# def readtxt():
#     root = '/home/jovyan/data-vol-1/wqg/lane/'
#     if os.path.isdir(root +"oringe"):
#         filelist = os.listdir(root +"oringe")
#         for fl in filelist:
#             imglist = os.listdir(os.path.join(root, "oringe",fl, 'image'))
#             for file in tqdm(imglist):
#                 if file[-3:] == 'jpg':
#                     txt = os.path.join("oringe",fl, 'image', file) + ' ' + os.path.join("oringe",fl, 'image', file[:-3] + 'png')
#                     info.append(txt + '\n')


if __name__=='__main__':
#     part_ = '935_936'
#     part_all = ['950','951','952','953','954','955','956','963','964','965','966','967','973','974','975','998']
#     part_all =['1777', '1722', '972', '971', '970', '969', '968', '962', '961', '960', '959', '958', '940', '939', '938', '937', '936', '935', '934', '933', '928', '925', '924', '923', '922', '892', '891', '630']
    part_all = ['1982']
    for part_ in part_all:
        pathdir = '/home/wqg/data/maxvision_data/line/'
        xmlpath = pathdir + part_ + '/annotations.xml'
        save_imgpath = pathdir +part_+'/mask_images'
        save_maskpath = pathdir +part_+'/gt_image'
        not_care_lable = '###'
        # if not os.path.exists(save_txtpath):
        os.makedirs(save_imgpath,exist_ok= True)
        os.makedirs(save_maskpath,exist_ok= True)
        # 使用minidom解析器打开 XML 文档
        DOMTree = xml.dom.minidom.parse(xmlpath)
        collection = DOMTree.documentElement

        # 在集合中获取所有图片的信息
        imgmessages = collection.getElementsByTagName("image")

        # 解析每一张图片
        # allImgmessage = []
        oneImg = datastru()
        # for messages_i in tqdm(range(len(imgmessages))):
        for messages_i in tqdm(range(100)):
            # messages_i =10
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


            type = messages.getElementsByTagName('polyline')
            str_label = ''

            label = np.zeros((352, 640), dtype=np.uint8)
            img = cv2.imread(os.path.join(pathdir,part_,'imageset',pathTXT))
            label = cv2.resize(img,dsize=(640,360))[8:,:]
            # cv2.imshow("label",label)
            # cv2.waitKey(0)

            for i in range(len(type)):
                label_point = {}
                try:
                    points = type[i].getAttribute("points").split(';')
                    value = type[i].getElementsByTagName('attribute')[0].firstChild.data
                    value = int(value)
                    # print(value)
                    if -3<value<0 :
                        value= value+3
                        draw(label, points, value, (1080, 1920), show=False)
                    elif 0 < value < 3:
                        value = value+2
                        draw(label, points, value, (1080,1920), show=False)
                except:
                    print(pathTXT)
                    # print()
            # name = "9%05d.jpg"%messages_i
            cv2.imwrite(os.path.join(pathdir, part_, 'mask_images', pathTXT), label)
            # cv2.imwrite(os.path.join(pathdir,part_,'gt_image',"9%05d.png"%messages_i),label)
            # cv2.imwrite(os.path.join(pathdir,part_,'image',"9%05d.jpg"%messages_i),img)
            
        print('end')

