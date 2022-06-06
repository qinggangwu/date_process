
import os
import shutil

import cv2
from tqdm import tqdm

from PIL import Image
from PIL import ImageDraw
import xml.etree.ElementTree as ET



def read_cavt():
    xmlpath = '/home/wqg/data/maxvision_data/segment/annotations.xml'

    imgdir = "/home/wqg/data/maxvision_data/segment/imageset"
    savedir = "/home/wqg/data/maxtest/"


    val = 1

    os.makedirs(os.path.join(savedir,"ann_dir", 'train', 'max'),exist_ok=True)
    os.makedirs(os.path.join(savedir,"img_dir", 'train', 'max'),exist_ok=True)

    tree = ET.ElementTree(file=xmlpath)
    root = tree.getroot()
    child_root = root[1:]
    for idx, child_of_image in tqdm(enumerate(child_root)):
        imginfo = child_of_image.attrib
        point =[]
        filename = "%06d"%idx

        size = (int(imginfo['width']) ,  int(imginfo['height']))      # ( annotation.imgWidth , annotation.imgHeight )
        labelImg = Image.new("L", size, 0)

        drawer = ImageDraw.Draw(labelImg)

        for child_of_point in child_of_image:
            point.append(child_of_point.attrib)
            points = child_of_point.attrib['points'].split(";")
            polygon =[ tuple(map(int, ii.split(',') )) for ii in points]

            # print(polygon)
            drawer.polygon(polygon, fill=val)

        jpgimg = cv2.imread(os.path.join(imgdir,imginfo['name']))
        cv2.imwrite( os.path.join(savedir,"img_dir", 'train', 'max',filename +'_leftImg8bit.png') ,jpgimg)
        # shutil.copy( os.path.join(savedir,"img_dir", 'train', 'max',filename +'_leftImg8bit.png') ,jpgimg)

        savefilename = os.path.join(savedir, "ann_dir", 'train', 'max',filename +'_gtFine_labelTrainIds_wu.png')
        labelImg.save(savefilename)

        # print('test');quit()

def chioce_cityspase():
    import mmcv
    import json
    oblist = ['car','truck', 'bus' ,'caravan','trailer']

    gt_dir = '/home/wqg/data/cityscapes/gtFine'
    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_polygons.json', recursive=True):
        poly_file = os.path.join(gt_dir, poly)

        with open(poly_file,'r') as pf:
            info = json.load(pf)
            objects = info["objects"]
            con = 0
            for obj in objects:
                if obj['label'] in oblist:
                    con +=1

            # print(objects);quit()

        if con > 13:
            # print(poly_file)
            poly_files.append(poly_file)
            pngname = poly_file.replace('_gtFine_polygons.json','_gtFine_labelTrainIds_wu.png')
            jpgname = poly_file.replace('_gtFine_polygons.json','_leftImg8bit.png').replace('gtFine','leftImg8bit')


            res = poly_file.split("/")[-2]
            savejpgname = jpgname.replace(res,"chioes",1).replace("cityscapes",'maxtest').replace('leftImg8bit','img_dir',1)
            savepngname = pngname.replace(res,"chioes",1).replace("cityscapes",'maxtest').replace('gtFine','ann_dir',1)

            shutil.copy(jpgname,savejpgname)
            shutil.copy(pngname,savepngname)

        print(len(poly_files))

if __name__ == "__main__":
    read_cavt()
    # chioce_cityspase()
    # print(list(map(lambda x: int(x),["2"] ) ))

    path = "/home/wqg/data/cityscapes/gtFine/train/aachen/aachen_000003_000019_gtFine_labelTrainIds.png"
    path0 = "/home/wqg/data/cityscapes/gtFine/train/aachen/aachen_000003_000019_gtFine_labelTrainIds_wu.png"

    # img = cv2.imread(path,0)
    # cv2.imshow('img',img)
    # cv2.imshow('img0', cv2.imread(path0,0))
    # cv2.waitKey(0)


    # dir = '/home/wqg/data/maxtest/img_dir/val/'
    #
    # file = ['max','chioes']
    # file = ['chioes']
    # info =[]
    # for fl in file:
    #     for name in os.listdir(dir +fl):
    #         jpg = 'val/'+ fl + "/"+name[:-16]
    #         info.append(jpg +'\n')
    #
    # with open('/home/wqg/data/maxtest/val.txt','w') as f:
    #     f.writelines(info)









