
import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

def analyze_xml(xmlPath):
    tree = ET.ElementTree(file=xmlPath)
    root = tree.getroot()
    child_root = root.findall('object')

    info_list = []
    for child_of_image in child_root:
        labelname = child_of_image.find('name').text
        # print('name',labelname)
        child_of_point = child_of_image.find('bndbox')
        xmin = child_of_point.find('xmin').text
        ymin = child_of_point.find('ymin').text
        xmax = child_of_point.find('xmax').text
        ymax = child_of_point.find('ymax').text

        info_list.append([labelname ,xmin,ymin,xmax,ymax])

        # print(xmin,ymin,xmax,ymax)
    return info_list

def make_process():
    rootdir = '/home/wqg/data/火灾烟雾数据'

    savedir ='/home/wqg/data/火灾烟雾数据'
    fileList = ['fire_img_label','VOC2020']

    savetrain_txtpath = os.path.join(savedir, 'det', 'train', 'labels')
    savetrain_imgpath = os.path.join(savedir, 'det', 'train', 'images')
    saveval_txtpath = os.path.join(savedir, 'det', 'valtest', 'labels')
    saveval_imgpath = os.path.join(savedir, 'det', 'valtest', 'images')

    os.makedirs(savetrain_txtpath, exist_ok=True)
    os.makedirs(savetrain_imgpath, exist_ok=True)
    os.makedirs(saveval_txtpath, exist_ok=True)
    os.makedirs(saveval_imgpath, exist_ok=True)


    labeldict = {'fire' :0 ,'smoke':1}
    for file in fileList:
        print(file)
        root = os.path.join(rootdir,file)

        file_list = os.listdir(os.path.join(root,'images'))[:30]
        for idx,imgname in tqdm(enumerate(file_list)):
            imgpath = os.path.join(root,'images',imgname)
            img = cv2.imread(imgpath)
            size = img.shape     # h,w,c

            xmlpath = os.path.join(root,'annotations',imgname[:-4] +'.xml')
            if os.path.exists(xmlpath):
                info_list = analyze_xml(xmlpath)
                labels = ''

                for linninfo in info_list:
                    labelnum = labeldict[linninfo[0]]
                    xtl = int(linninfo[1])
                    ytl = int(linninfo[2])
                    xbr = int(linninfo[3])
                    ybr = int(linninfo[4])

                    width = xbr - xtl
                    height = ybr - ytl
                    center_x = xtl + int(0.5 * width)
                    center_y = ytl + int(0.5 * height)

                    if center_x < 0 or center_y < 0 or center_x > size[1] or center_y > size[0]:
                        print('center_x,center_y', center_x, center_y)
                        print(xbr, xtl, ybr, ytl)

                    sing_label = "{} {} {} {} {}\n".format(labelnum,
                                                           round(center_x / size[1], 6),
                                                           round(center_y / size[0], 6),
                                                           round(width / size[1], 6),
                                                           round(height / size[0], 6))
                    labels += sing_label

            if idx%10 == 0 :
                new_imgpath = os.path.join(saveval_imgpath , imgname)
                new_labelpath = os.path.join(saveval_txtpath , imgname[:-3] + 'txt')

                cv2.imwrite(new_imgpath,img)

                with open(new_labelpath, "w", encoding='utf-8') as f:
                        f.write(labels)

            else:
                new_imgpath = os.path.join(savetrain_imgpath, imgname)
                new_labelpath = os.path.join(savetrain_txtpath, imgname[:-3] + 'txt')
                cv2.imwrite(new_imgpath, img)

                with open(new_labelpath, "w", encoding='utf-8') as f:
                    f.write(labels)


def main():
    dir_root = '/home/wqg/data/火灾烟雾数据/fire_img_label'
    make_process(dir_root)



if __name__ == "__main__":
    # main()
    make_process()