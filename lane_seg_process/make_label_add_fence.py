

import cv2
import os
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm


class Precess:
    def __init__(self,root):
        self.root = root


    def get_xmlinfo(self,xmlpath):

        tree = ET.ElementTree(file=xmlpath)
        root = tree.getroot()
        child_root = root[1:]

        info_list = []
        for child_of_image in child_root:
            imginfodict = child_of_image.attrib

            pointlist = [[[int(iii) for iii in ii.split(',')] for ii in child_of_point.attrib['points'].split(';')] for child_of_point in child_of_image]
            info_list.append((imginfodict, pointlist))

        return info_list


    def add_fence(self):
        xmlpath = os.path.join(self.root, 'annotations.xml')
        infodict = self.get_xmlinfo(xmlpath)

        for idx,info in tqdm(enumerate(infodict)):
            infoimgdict = info[0]

            num = int(infoimgdict['name'][-10:-4])
            file = str(num//500)
            labelname = '%06d.png'%num
            labelpath = os.path.join(self.root,'gt_imgs',file , labelname)

            savepath = os.path.join(self.root,'test_gt_imgs',file )
            os.makedirs(savepath,exist_ok= True)



            imgpath = os.path.join(self.root,'imageset',infoimgdict['name'])
            img = cv2.imread(labelpath)

            # 修改标签
            for ploy in info[1]:
                ploy0 = np.array(ploy)
                img = cv2.fillConvexPoly(img,ploy0,(100,50,0))
                # img = cv2.fillPoly(img,[ploy0],(100,50,0))


            cv2.imwrite(os.path.join(savepath , labelname),img)

            # cv2.imshow('img', cv2.resize(img, dsize=None, fx=0.3, fy=0.3))
            # cv2.waitKey(0)


def main():

    root = '/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/data/apolloscape/2044'
    P = Precess(root)

    P.add_fence()



if __name__ == "__main__":
    main()








