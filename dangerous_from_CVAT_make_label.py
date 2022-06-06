

import cv2
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm



class Process:
    def __init__(self,root,savedir):
        self.root = root
        self.savedir = savedir
        self.fileLise = ['危险品车']

        self.labeldict = {'危险品车':0 , '危险标识':1}


    def make_savefile(self):
        file = 'det'

        self.savetrain_txtpath = os.path.join(self.savedir, file, 'train', 'labels')
        self.savetrain_imgpath = os.path.join(self.savedir, file, 'train', 'images')
        self.saveval_txtpath = os.path.join(self.savedir, file, 'valtest', 'labels')
        self.saveval_imgpath = os.path.join(self.savedir, file, 'valtest', 'images')

        os.makedirs(self.savetrain_txtpath, exist_ok=True)
        os.makedirs(self.savetrain_imgpath, exist_ok=True)
        os.makedirs(self.saveval_txtpath, exist_ok=True)
        os.makedirs(self.saveval_imgpath, exist_ok=True)

    def process_xml(self,xmlpath):
        info = []

        # xmlpath = os.path.join(data_root, part, 'annotations.xml')
        tree = ET.ElementTree(file=xmlpath)
        root = tree.getroot()
        child_root = root

        for idx, child_of_image in enumerate(child_root):
            imginfo = child_of_image.attrib
            point = [child_of_point.attrib for child_of_point in child_of_image ]

            if len(point)==0 or len(point[0])==0:
                continue

            # if len(point) <= 1:
            #     print('point',point)

            info.append( [imginfo , point] )

        return info

    def save_label(self,saveimgpath,img ,info):
        hzm = os.path.splitext(saveimgpath)[-1]
        savetxtpath = saveimgpath.replace('images','labels').replace(hzm,".txt")

        cv2.imwrite(saveimgpath,img)

        with open(savetxtpath, 'w', encoding='utf-8') as  f:
            f.writelines(info)




    def process_label(self):

        self.make_savefile()

        for file in self.fileLise:

            xmlpath = os.path.join(self.root , file, 'annotations.xml')
            imgpath = os.path.join(self.root , file, 'imageset')
            xmlpath =  '/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/data/maxvision_data/危险品车_annotations.xml'

            infoList = self.process_xml(xmlpath)

            crop_num = 0


            for idx ,info in tqdm(enumerate(infoList)):
                imginfo = info[0]

                if imginfo['name'] != '-nfs-车辆违规运输-危险品车-危险品车-c4a357b5b33056a70521cd78148b1b21.jpg':
                    continue


                size = (int(imginfo['width']),int(imginfo['height']))    #  (x ,y)
                img = cv2.imread(os.path.join(imgpath,imginfo['name']))


                car_num = len([label for label in info[1] if label['label'] == '危险品车'] )

                labelall = ''
                for piont in info[1]:
                    if piont['label'] == '车牌':
                        continue

                    xbr = int(piont['xbr']) if int(piont['xbr']) < size[0] else size[0]
                    xtl = int(piont['xtl']) if int(piont['xtl']) >0 else 0
                    ybr = int(piont['ybr']) if int(piont['ybr']) < size[1] else size[1]
                    ytl = int(piont['ytl']) if int(piont['ytl']) >0 else 0

                    width = xbr - xtl
                    height = ybr - ytl
                    center_x = xtl + int(0.5 * width)
                    center_y = ytl + int(0.5 * height)

                    if center_x < 0 or center_y < 0 or center_x > size[0] or center_y > size[1]:
                        print('center_x,center_y', center_x, center_y)
                        # print(xbr, xtl, ybr, ytl)

                    sing_label1 = "{} {} {} {} {}\n".format(self.labeldict[piont['label']],
                                                           round(center_x / size[0], 6),
                                                           round(center_y / size[1], 6),
                                                           round(width / size[0], 6),
                                                           round(height / size[1], 6))

                    labelall += sing_label1



                    # 保存标签和图片

                    try:

                        if piont['label'] == '危险品车' and car_num == 1:
                            pad = 4
                            crop_xbr = int(piont['xbr']) + pad if int(piont['xbr']) + pad < size[0]  else size[0]
                            crop_xtl = int(piont['xtl']) - pad if int(piont['xtl']) - pad >0 else 0
                            crop_ybr = int(piont['ybr']) + pad if int(piont['ybr']) + pad < size[1] else size[1]
                            crop_ytl = int(piont['ytl']) - pad if int(piont['ytl']) - pad >0 else 0


                            crop_img = img[crop_ytl: crop_ybr, crop_xtl: crop_xbr]
                            # crop_img = img[ytl: ybr, xtl: xbr]
                            # cropImg_size = (width + 2 * pad ,height + 2* pad )
                            cropImg_size = (crop_xbr -  crop_xtl,crop_ybr - crop_ytl )

                            corp_label = ''

                        elif piont['label'] == '危险标识' and car_num == 1:

                            if len(crop_img) == 0:
                                continue

                            cxbr = int(piont['xbr']) -crop_xtl if int(piont['xbr']) - crop_xtl < cropImg_size[0] else cropImg_size[0]
                            cxtl = int(piont['xtl']) -crop_xtl if int(piont['xtl']) - crop_xtl >0 else 0
                            cybr = int(piont['ybr']) -crop_ytl if int(piont['ybr']) - crop_ytl < cropImg_size[1] else cropImg_size[1]
                            cytl = int(piont['ytl']) -crop_ytl if int(piont['ytl']) - crop_ytl >0 else 0


                            # 画框
                            # cv2.rectangle(crop_img , (cxtl, cytl), (cxbr, cybr), (0, 255, 0), 2)

                            cwidth = cxbr - cxtl
                            cheight = cybr - cytl
                            ccenter_x = cxtl + int(0.5 * cwidth)
                            ccenter_y = cytl + int(0.5 * cheight)

                            if ccenter_x < 0 or ccenter_y < 0 or ccenter_x > cropImg_size[0] or ccenter_y > cropImg_size[1]:
                                print('cropimg center_x,center_y', ccenter_x, ccenter_y,cropImg_size)
                                # print(cxbr, cxtl, cybr, cytl)
                                print(imginfo['name'])

                            csing_label = "{} {} {} {} {}\n".format(self.labeldict[piont['label']],
                                                                   round(ccenter_x / cropImg_size[0], 6),
                                                                   round(ccenter_y / cropImg_size[1], 6),
                                                                   round(cwidth / cropImg_size[0], 6),
                                                                   round(cheight / cropImg_size[1], 6))

                            corp_label += csing_label

                            # print(csing_label)
                            if piont == info[1][-1]:
                            #
                            #     imgsavepath = os.path.join(self.root , 'crop_test', "crop_"+imginfo['name'])
                            #     os.makedirs(os.path.join(self.root , 'crop_test'),exist_ok=True)
                            #
                            #     cv2.imwrite(imgsavepath,crop_img)



                                # 保存corp图片
                                if crop_num % 10 == 0:
                                    imgsavepath = os.path.join(self.saveval_imgpath, "crop_"+imginfo['name'])
                                    self.save_label(imgsavepath, img, corp_label)
                                    crop_num +=1
                                else:
                                    imgsavepath = os.path.join(self.savetrain_imgpath, "crop_"+imginfo['name'])
                                    self.save_label(imgsavepath, img, corp_label)
                                    crop_num +=1
                    except:
                        print(imginfo['name'])


                # 保存 原始 图片
                if idx %10 ==0:
                    imgsavepath = os.path.join(self.saveval_imgpath,imginfo['name'] )
                    self.save_label(imgsavepath, img, labelall)
                else:
                    imgsavepath =  os.path.join( self.savetrain_imgpath ,imginfo['name'] )
                    self.save_label(imgsavepath ,img,labelall)





def main():
    root = '/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/data/maxvision_data/'
    savedir = '/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/data/maxvision_data/'
    P = Process(root,savedir)



    xmlpath = '/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/data/maxvision_data/危险品车_annotations.xml'

    P.process_label()

    # info = P.save_label(xmlpath)

    # print(info[0])
    # print(len(info))



    pass



if __name__ == "__main__":
    main()