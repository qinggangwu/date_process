import os
import random
import shutil

import numpy as np
import cv2
from collections import namedtuple
from copy import copy
from tqdm import tqdm

# import torch
# from torch.utils.data import Dataset,DataLoader  # 导入Dataset后可以使用“help(Dataset)查看官方文档”
# from PIL import Image
# import matplotlib.pyplot as plt

def getLabe():
    Label = namedtuple('Label', [

        'name',  # The identifier of this label.
        # We use them to uniquely name a class

        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.

        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!

        'category',  # The name of the category that this label belongs to

        'categoryId',  # The ID of this category. Used to create ground truth images
        # on category level.

        'hasInstances',  # Whether this label distinguishes between single instances or not

        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not

        'color',  # The color of this label
    ])

    labels = [
                #           name     id     trainId      category  catId   hasInstances  ignoreInEval    color
                # Label(     'void' ,   0 ,     0,        'void' ,   0 ,      False ,      False , (  0,   0,   0) ),

                Label(   'om_n_n' , 250 ,    0 ,     'others' ,  11 ,      False ,      False , (102,   0, 204) ),
                Label(    'b_w_g' , 201 ,     1 ,    'guiding' ,   2 ,      False ,      False , (  0,   0, 142) ),   # 白虚线
                Label(    'b_y_g' , 203 ,     2 ,    'guiding' ,   2 ,      False ,      False , (119,  11,  32) ),   # 黄虚线
                Label(    's_w_d' , 200 ,     3 ,   'dividing' ,   1 ,      False ,      False , ( 70, 130, 180) ),  # 白实线
                Label(    's_y_d' , 204 ,     4 ,   'dividing' ,   1 ,      False ,      False , (220,  20,  60) ),  # 黄实线
                Label(  'ds_y_dn' , 209 ,     4 ,   'dividing' ,   1 ,      False ,      False , (255, 0,   0) ),   # 双 黄 线

                Label(    's_w_s' , 217 ,    6 ,   'stopping' ,   3 ,      False ,      False , (220, 220,   0) ),   # 停止线
                Label(  'r_wy_np' , 227 ,    7 , 'no parking' ,  10 ,      False ,      False , (178, 132, 190) ),

                Label(    'a_w_t' , 220 ,    5 ,  'thru/turn' ,   7 ,      False ,      False , (128, 128,   0) ),
                Label(   'a_w_tl' , 221 ,    5 ,  'thru/turn' ,   7 ,      False ,      False , (128,  78, 160) ),
                Label(   'a_w_tr' , 222 ,    5 ,  'thru/turn' ,   7 ,      False ,      False , (150, 100, 100) ),
                Label(    'a_w_l' , 224 ,    5 ,  'thru/turn' ,   7 ,      False ,      False , (180, 165, 180) ),
                Label(    'a_w_r' , 225 ,    5 ,  'thru/turn' ,   7 ,      False ,      False , (107, 142,  35) ),
                Label(   'a_w_lr' , 226 ,    5 ,  'thru/turn' ,   7 ,      False ,      False , (201, 255, 229) ),

                Label(   'b_n_sr' , 205 ,    0 ,  'reduction' ,   8 ,      False ,      False , (255, 128,   0) ),
                Label(    's_w_p' , 210 ,    0 ,    'parking' ,   5 ,      False ,      False , (128,  64, 128) ),  # 停车线
                Label(   'c_wy_z' , 214 ,    0 ,      'zebra' ,   6 ,      False ,      False , (190, 153, 153) ),   # 斑马线
                Label(   'f_wy_z' , 214 ,    8 ,      'fence' ,   6 ,      False ,      False , (100, 50, 0) ),   # 栏栅




                # Label(  'sb_w_do' , 206 ,     5 ,   'dividing' ,   1 ,      False ,       True , (  0,   0,  60) ),  # 模糊双白线
                # Label(  'sb_y_do' , 207 ,     6 ,   'dividing' ,   1 ,      False ,       True , (  0,  60, 100) ),  # 模糊双黄线
                # Label(  'ds_w_dn' , 213 ,     3 ,   'dividing' ,   1 ,      False ,       True , (128,   0, 128) ),
                # Label(   'db_w_g' , 211 ,     0 ,    'guiding' ,   2 ,      False ,       True , (244,  35, 232) ),
                # Label(   'db_y_g' , 208 ,    0 ,    'guiding' ,   2 ,      False ,       True , (  0,   0, 160) ),
                # Label(   'db_w_s' , 216 ,    7 ,   'stopping' ,   3 ,      False ,       True , (153, 153, 153) ),
                # Label(   'ds_w_s' , 215 ,    7 ,   'stopping' ,   3 ,      False ,       True , (250, 170,  30) ),
                # Label(    's_w_c' , 218 ,    14 ,    'chevron' ,   4 ,      False ,       True , (102, 102, 156) ),
                # Label(    's_y_c' , 219 ,    15 ,    'chevron' ,   4 ,      False ,       True , (128,   0,   0) ),
                # Label(    's_n_p' , 232 ,    17 ,    'parking' ,   5 ,      False ,       True , (238, 232, 170) ),
                # Label(    'a_w_u' , 202 ,    19 ,  'thru/turn' ,   7 ,      False ,       True , (  0,   0, 230) ),
                # Label(  'a_w_tlr' , 231 ,    23 ,  'thru/turn' ,   7 ,      False ,       True , (255, 165,   0) ),
                # Label(   'a_n_lu' , 230 ,    27 ,  'thru/turn' ,   7 ,      False ,       True , (0,   191, 255) ),
                # Label(   'a_w_tu' , 228 ,    28 ,  'thru/turn' ,   7 ,      False ,       True , ( 51, 255,  51) ),
                # Label(    'a_w_m' , 229 ,    29 ,  'thru/turn' ,   7 ,      False ,       True , (250, 128, 114) ),
                # Label(    'a_y_t' , 233 ,    30 ,  'thru/turn' ,   7 ,      False ,       True , (127, 255,   0) ),
                # Label(  'd_wy_za' , 212 ,    32 ,  'attention' ,   9 ,      False ,       True , (  0, 255, 255) ),
                # Label( 'vom_wy_n' , 223 ,     0 ,     'others' ,  11 ,      False ,       True , (128, 128,  64) ),
                # Label(    'noise' , 249 ,   255 ,    'ignored' , 255 ,      False ,       True , (  0, 153, 153) ),
                # Label(  'ignored' , 255 ,   255 ,    'ignored' , 255 ,      False ,       True , (255, 255, 255) ),
            ]

    return labels
def get_chioce_Labe():
    Label = namedtuple('Label', [ 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color',])

    labels = [
                Label(    'b_y_g' , 203 ,     2 ,    'guiding' ,   2 ,      False ,      False , (119,  11,  32) ),   # 黄虚线
                Label(  'r_wy_np' , 227 ,    7 , 'no parking' ,  10 ,      False ,      False , (178, 132, 190) ),
            ]

    return labels

def get_file_list(path,imgList=[],suffix = ['.jpg','.png']):

    for f in os.listdir(path):
        flie = os.path.join(path, f)

        if os.path.isfile(flie) and os.path.splitext(flie)[-1] in suffix:
            imgList.append(flie)
        elif os.path.isfile(flie) and os.path.splitext(flie)[-1] not in suffix:
            continue
        else:
            get_file_list(flie,imgList,suffix)
    return imgList

class ApolloScape_data_process():
    def __init__(self,root):
        self.root = root
        self.imgList = []
        self.labelList = []
        self.labels = getLabe()
        self.bs = 2


    def get_img_label_list(self):
        '/home/wqg/data/apolloscap/ColorImage_road02/ColorImage/Record001'

        root = self.root
        froadList  = ['ColorImage_road02','ColorImage_road03','ColorImage_road04']
        # froadList  = ['road03','road03']
        CameraList = ['Camera 5','Camera 6']
        for ind in range(len(froadList)):
            fr = froadList[ind]
            for f in os.listdir(os.path.join(root,fr,'ColorImage')):
                for ff in CameraList:
                    for filename in os.listdir(os.path.join(root,fr,'ColorImage',f,ff)):
                        if filename[-3:] != 'jpg':
                            continue

                        imgpath = os.path.join(root,fr,'ColorImage',f,ff,filename)

                        labelname = filename[:-4] + '_bin.png'
                        fr_label = fr.replace('ColorImage','Labels')
                        labepath = os.path.join(root,fr_label,'Label',f,ff,labelname)

                        if os.path.isfile(imgpath) and os.path.isfile(labepath):
                            self.imgList.append(imgpath)
                            self.labelList.append(labepath)

    def get_chioce_list(self):
        '/home/wqg/data/apolloscap/ColorImage_road02/ColorImage/Record001'

        root = self.root
        imgfilename = 'choice_Y/imgs'
        pngfilename = 'choice_Y/gt_imgs'


        # file_list = [str(i) for i in range(32)]
        file_list = os.listdir(os.path.join(root,imgfilename))

        for fi in file_list:
            for filename in os.listdir(os.path.join(root,  imgfilename, fi)):
                if filename[-3:] != 'jpg':
                    continue
                imgpath = os.path.join(root,  imgfilename, fi, filename)

                labelname = filename[:-4] + '.png'
                # fr_label = fr.replace('ColorImage', 'Labels')
                labepath = os.path.join(root,  pngfilename, fi, labelname)

                if os.path.isfile(imgpath) and os.path.isfile(labepath):
                    self.imgList.append(imgpath)
                    self.labelList.append(labepath)

        # froadList = ['ColorImage_road03', 'ColorImage_road04']
        # # froadList  = ['road03','road03']
        # CameraList = ['Camera 5', 'Camera 6']
        # for ind in range(len(froadList)):
        #     fr = froadList[ind]
        #     for f in os.listdir(os.path.join(root,fr,'ColorImage')):
        #         for ff in CameraList:
        #             for filename in os.listdir(os.path.join(root,fr,'ColorImage',f,ff)):
        #                 if filename[-3:] != 'jpg':
        #                     continue
        #
        #
        #                 imgpath = os.path.join(root,fr,'ColorImage',f,ff,filename)
        #
        #                 labelname = filename[:-4] + '_bin.png'
        #                 fr_label = fr.replace('ColorImage','Labels')
        #                 labepath = os.path.join(root,fr_label,'Label',f,ff,labelname)
        #
        #                 if os.path.isfile(imgpath) and os.path.isfile(labepath):
        #                     self.imgList.append(imgpath)
        #                     self.labelList.append(labepath)

    def make_mask(self):

        color2label = {label.color: label for label in self.labels}

        os.makedirs(self.root +'/test/gt_imgs',exist_ok=True )
        os.makedirs(self.root +'/test/imgs',exist_ok=True )

        for i in tqdm(range(1,1000,2)):
            labelimg = cv2.imread(self.labelList[i])
            img = cv2.imread(self.imgList[i])

            shape = (720,1280)

            img = cv2.resize(img[807:,:],dsize=(1280,720))
            # labelimg = cv2.resize(labelimg[807:,:],dsize=shape)
            labelimg = labelimg[807:,:]

            # newImge = np.ones(shape) *255
            newImge = np.zeros(shape)


            for cl in color2label:
                # ccN = np.array(cl)
                # ccN = cl[0]* 0.1 + cl[1]* 0.4 + cl[2]* 0.5

                r = copy( labelimg[:, :, 0])
                g = copy( labelimg[:, :, 1])
                b = copy( labelimg[:, :, 2])

                r[ np.where( r == cl[2])] =1
                g[ np.where(g == cl[1])] =1
                b[ np.where(b == cl[0])] =1

                rgb = r + g +b
                point = np.where(rgb == 3)

                if point[0].size == 0 :
                    continue


                # if cl == (255, 0,   0):
                #     print(self.labelList[i] , cl)


                X = np.array(point[0]*(720 / 1903)).astype(int)
                Y = np.array(point[1]*(1280 / 3384)).astype(int)

                # print('X size  : ',X.size)
                # print('categoryId',color2label[cl].categoryId * 20)
                newImge[(X ,Y)] = color2label[cl].trainId

                # point = labelimg(np.where(labelimg == cc))

                # print(np.max(point[1]))
                # print(point)

            # cv2.namedWindow("labelimg", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # cv2.imshow('img', newImge)
            # cv2.imshow('labelimg', labelimg)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            cv2.imwrite("/home/wqg/data/apolloscape/test/gt_imgs/%06d.png"%i,newImge)
            cv2.imwrite("/home/wqg/data/apolloscape/test/imgs/%06d.jpg"%i,img)

    def make_train_txt(self):

        train_info = []
        val_info = []
        filename = "test/imgs/"

        fileList = os.listdir(self.root + filename)
        for idx,name in enumerate(fileList):
            if name[-3:] != 'jpg':
                continue
            # pngname = name[:-4] +'.png'

            fo = name[:-4] + '\n'

            if idx%10 ==0:

                val_info.append(fo)
            else:
                train_info.append(fo)

        with open(self.root +"train.txt",'w') as tr:
            tr.writelines(train_info)
        with open(self.root + "val.txt", 'w') as va:
            va.writelines(val_info)

    def choice_pic(self):

        chioce_labels = get_chioce_Labe()

        color2label = {label.color: label for label in chioce_labels}

        os.makedirs(self.root +'/choice_all/gt_imgs',exist_ok=True )
        os.makedirs(self.root +'/choice_all/imgs',exist_ok=True )
        chioce_num = 0


        for i in tqdm(range(1,len(self.labelList))):
            labelimg = cv2.imread(self.labelList[i])
            # img = cv2.imread(self.imgList[i])

            for cl in color2label:
                try:
                    r = copy( labelimg[:, :, 0])
                    g = copy( labelimg[:, :, 1])
                    b = copy( labelimg[:, :, 2])

                    r[ np.where( r == cl[2])] =1
                    g[ np.where(g == cl[1])] =1
                    b[ np.where(b == cl[0])] =1

                    rgb = r + g +b
                    point = np.where(rgb == 3)

                    if point[0].size <= 20 :
                        continue
                    else:
                        chioce_num +=1

                        # 移动目标图片
                        filename = str(chioce_num // 500)
                        os.makedirs(os.path.join( self.root,'choice_all/imgs' ,filename) ,exist_ok=True)
                        os.makedirs(os.path.join( self.root,'choice_all/gt_imgs' ,filename) ,exist_ok=True)
                        imgname = self.imgList[i].split('/')[-1]
                        pngname = self.labelList[i].split('/')[-1]

                        img_newpath = os.path.join( self.root,'choice_all/imgs' ,filename, imgname)
                        png_newpath = os.path.join( self.root,'choice_all/gt_imgs' ,filename, pngname)
                        # print('end')
                        shutil.copy(self.imgList[i],img_newpath)
                        shutil.copy(self.labelList[i],png_newpath)
                        break
                except:
                    print(self.labelList[i])
            if i %300 == 1:
                print(chioce_num)

    def get_percent(self):
        percent_dict = {label.trainId : 0 for label in self.labels}
        color2label = {label.color: label for label in self.labels}
        n = 1

        for i in tqdm(range(n,len(self.labelList),3)):
        # for i in tqdm(range(10)):
            labelimg = cv2.imread(self.labelList[i])

            for cl in color2label:
                r = copy( labelimg[:, :, 0])
                g = copy( labelimg[:, :, 1])
                b = copy( labelimg[:, :, 2])

                r[ np.where( r == cl[2])] =1
                g[ np.where(g == cl[1])] =1
                b[ np.where(b == cl[0])] =1

                rgb = r + g +b
                point = np.where(rgb == 3)

                if point[0].size == 0 :
                    continue
                percent_dict[color2label[cl].trainId] += len(point[0])
                # print(color2label[cl].trainId ,len(point[0]))

            if i %50 ==0:
                print(i, percent_dict)


    def moive_errer_img(self):

        save_error_path = '/home/wqg/data/apolloscape/choice_Y/errorfile/'
        os.makedirs(save_error_path,exist_ok=True)


        erfile_path = '/home/wqg/data/apolloscape/choice/error'

        # er_imgfile_list = os.listdir(erfile_path)
        er_imgname_list = [name[:-4] for name in os.listdir(erfile_path)]

        for ind,imgname in enumerate(self.imgList):
            filename = imgname.split('/')[-1][:-4]
            if filename in er_imgname_list:

                newimgpath =save_error_path + filename +'.jpg'
                newpngpath =save_error_path + filename +'.png'

                shutil.move(self.imgList[ind],newimgpath)
                shutil.move(self.labelList[ind],newpngpath)
                # print(filename)
                # print(self.imgList[ind],self.labelList[ind]);quit()


        # print(er_imgname_list)









def get_lane_seg_precent():

    dit1 = {0: 253576664, 1: 70916059, 2: 0, 3: 194646097, 4: 47781985, 6: 21626119, 7: 41951, 5: 81213074 ,'pic': 5800 }
    dit2 = {0: 257218871, 1: 71112221, 2: 0, 3: 195062715, 4: 47671666, 6: 22284219, 7: 48954, 5: 81846767 ,'pic': 5850 }
    dit3 = {0: 255395186, 1: 70686615, 2: 0, 3: 194070245, 4: 47798858, 6: 22075820, 7: 48840, 5: 80538366 ,'pic': 5800}
    dit4 = {0: 72842020, 1: 17456135, 2: 0, 3: 40813597, 4: 19407350, 6: 6204421, 7: 11904, 5: 28320504  ,'pic': 4550}
    dit5 = {0: 252581361, 1: 71137707, 2: 0, 3: 193898500, 4: 47977817, 6: 20982868, 7: 51515, 5: 80220686 ,'pic': 5800}
    dit6 = {0: 254010995, 1: 70704893, 2: 0, 3: 194252769, 4: 47947820, 6: 21032399, 7: 54893, 5: 80343098 ,'pic': 5800}
    dit7 = {0: 251212210, 1: 70072638, 2: 0, 3: 193639504, 4: 47159395, 6: 21004094, 7: 58391, 5: 79060906 ,'pic': 5750}
    dit8 = {0: 250269442, 1: 70384630, 2: 0, 3: 193560231, 4: 46921801, 6: 20804509, 7: 62919, 5: 79277595,'pic': 5750}
    dit9 = {0: 255509564, 1: 71619023, 2: 0, 3: 195425690, 4: 47868966, 6: 20944306, 7: 51636, 5: 82470326,'pic': 5850}
    dit10 = {0: 249803138, 1: 70552034, 2: 0, 3: 193578806, 4: 46358541, 6: 20545441, 7: 45006, 5: 79619427,'pic': 5700}

    dit11 = {0: 131797533, 1: 40936255, 2: 0, 3: 135368548, 4: 13365771, 6: 9180789, 7: 42211, 5: 27426595 }
    dit12 = {}

    namedict = {0: '其他类别', 1: '白虚线', 2: '黄虚线', 3: '白实线', 4: '黄实线', 5: '导流线', 6: '停止线', 7: '禁停区', 8: '栏栅'}
    dictsum = {}

    dictlist = [dit1,dit2,dit3,dit4,dit5,dit6,dit7,dit8,dit9,dit10,dit11]
    for di in dictlist:
        if di is None:
            continue

        for key in di:
            val = di[key]
            if key in dictsum.keys():
                dictsum[key] +=val
            else:
                dictsum[key] = val

    print(dictsum)

    del dictsum['pic']
    # del dictsum[0]

    dictprecent = {}

    sums = sum(dictsum.values())
    for key in dictsum:
        name = namedict[key]
        dictprecent[name] = round(dictsum[key]/sums * 100, 4 )


    print(dictprecent)



if __name__ =="__main__":
    """
    ColorImage_road02    33119 
    """

    path = '/home/wqg/data/apolloscape/'
    P = ApolloScape_data_process(path)
    # P.get_img_label_list()     # 获取所有文件路径
    P.get_chioce_list()     # 获取所有选择文件路径

    P.moive_errer_img()  # 剔除所有错误图片


    # n = random.randint(0,15000)
    # print(P.imgList[n])
    # print(P.labelList[n])

    # P.make_mask()     # 修改图片标签

    # P.get_percent()

    # P.choice_pic()      # 挑选图片
    # get_lane_seg_precent()

    # P.make_train_txt()
    print('end');quit()




    pp = '/home/wqg/data/apolloscape/Labels_road02/Label/Record042/Camera 5/170927_073442476_Camera_5_bin.png'
    pp = '/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 5/170927_063949364_Camera_5_bin.png'

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    img = cv2.imread(pp)
    cv2.imshow('img', img)
    cv2.waitKey(0)

"""
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 5/170927_064010293_Camera_5_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 5/170927_064004199_Camera_5_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 5/170927_063955298_Camera_5_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 5/170927_063944979_Camera_5_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 5/170927_063949364_Camera_5_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 5/170927_063958007_Camera_5_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 6/170927_064011944_Camera_6_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 6/170927_064002910_Camera_6_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 6/170927_063937197_Camera_6_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 6/170927_063946399_Camera_6_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 6/170927_063943690_Camera_6_bin.png (0, 0, 230)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071221227_Camera_5_bin.png (0, 0, 60)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071221227_Camera_5_bin.png (0, 60, 100)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071153134_Camera_5_bin.png (0, 0, 60)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071153134_Camera_5_bin.png (0, 60, 100)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071153134_Camera_5_bin.png (128, 128, 64)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071203266_Camera_5_bin.png (0, 0, 60)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071203266_Camera_5_bin.png (128, 128, 64)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071144076_Camera_5_bin.png (0, 0, 60)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071144076_Camera_5_bin.png (0, 191, 255)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071144076_Camera_5_bin.png (128, 128, 64)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071213695_Camera_5_bin.png (0, 0, 60)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071213695_Camera_5_bin.png (0, 60, 100)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 5/170927_071213695_Camera_5_bin.png (128, 128, 64)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 6/170927_071214291_Camera_6_bin.png (0, 0, 60)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 6/170927_071214291_Camera_6_bin.png (0, 60, 100)
/home/wqg/data/apolloscape/Labels_road02/Label/Record026/Camera 6/170927_071214291_Camera_6_bin.png (128, 128, 64)



/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 5/170927_063944979_Camera_5_bin.png (178, 132, 190)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 5/170927_063949364_Camera_5_bin.png (178, 132, 190)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 6/170927_063946399_Camera_6_bin.png (178, 132, 190)
/home/wqg/data/apolloscape/Labels_road02/Label/Record002/Camera 6/170927_063943690_Camera_6_bin.png (178, 132, 190)


"""
