import os
import cv2
# from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image


def change_txt():
    ss = []
    with open(path,'r') as f:
        infoList = f.readlines()
    for info in infoList:
        info = info.replace('lane','lanenew')
        ss.append(info)
    with open(path,'w') as r:
        r.writelines(ss)

def tf_resize(path):

    img = Image.open(path)
    data_resize = transforms.Compose([transforms.Resize((256, 512))])
    imgre = data_resize(img)

    newpath = path.replace("lane",'lanenew')
    # os.makedirs('/home/wqg/data/lanenew/CurveLanes/train/gt_binary_image/')
    imgre.save(newpath)

def main(dir_root):
    fileList = ['tusimple','CurveLanes','CULane']
    for file in fileList:
        for part in ['train','val']:
            os.makedirs(os.path.join(dir_root.replace("lane",'lanenew'),file,part,'gt_image'), exist_ok=True)
            os.makedirs(os.path.join(dir_root.replace("lane",'lanenew'),file,part,'gt_binary_image'), exist_ok=True)
            os.makedirs(os.path.join(dir_root.replace("lane",'lanenew'),file,part,'gt_instance_image'), exist_ok=True)
            print(os.path.join(dir_root,file,part,'gt_image'))
            for imgname in tqdm(os.listdir(os.path.join(dir_root,file,part,'gt_image'))[:100]):
                pngname = imgname.replace('jpg','png')
                tf_resize(os.path.join(dir_root,file,part,'gt_image',imgname))
                tf_resize(os.path.join(dir_root,file,part,'gt_binary_image',pngname))
                tf_resize(os.path.join(dir_root,file,part,'gt_instance_image',pngname))

def resize_culane(dirroot):
    shape = (352,640,3)

    imglist = os.listdir(os.path.join(dirroot,'images'))
    for imgname in tqdm(imglist):
        if os.path.splitext(imgname)[-1] != '.jpg':
            continue
        imgpath = os.path.join(dirroot,'images',imgname)
        maskpath = os.path.join(dirroot,'gt_image',imgname.replace('jpg','png'))

        img = cv2.imread(imgpath)
        mask = cv2.imread(maskpath)
        maskbg = np.zeros(shape,dtype=np.uint8)
        imgbg = np.zeros(shape,dtype=np.uint8)

        maskbg[0:230,0:640,:] = cv2.resize(mask,dsize=(640,230))
        imgbg[0:230,0:640,:] = cv2.resize(img,dsize=(640,230))


        # cv2.imshow("img",imgbg)
        # cv2.imshow("mask",maskbg)
        cv2.imwrite(imgpath,imgbg)
        cv2.imwrite(maskpath,maskbg)

def resize_tusimple(dirroot):
    # shape = (352,640,3)
    reshape = (640,360)

    imglist = os.listdir(os.path.join(dirroot,'images'))
    for imgname in tqdm(imglist):
        if os.path.splitext(imgname)[-1] != '.jpg':
            continue
        imgpath = os.path.join(dirroot,'images',imgname)
        maskpath = os.path.join(dirroot,'gt_image',imgname.replace('jpg','png'))

        if os.path.isfile(imgpath) and os.path.isfile(maskpath):
            img =  cv2.resize(cv2.imread(imgpath),dsize= reshape)
            mask = cv2.resize(cv2.imread(maskpath),dsize= reshape)

            mask = mask[8:360,0:640,:]
            img = img[8:360,0:640,:]

            cv2.imwrite(imgpath,img)
            cv2.imwrite(maskpath,mask)

def resize_CurveLanes(dirroot):
    # shape = (352,640,3)
    reshape = (640,360)
    os.makedirs(os.path.join(dirroot,'images_resize'), exist_ok=True )

    imglist = os.listdir(os.path.join(dirroot,'images'))
    for imgname in tqdm(imglist):
        if os.path.splitext(imgname)[-1] != '.jpg':
            continue
        imgpath = os.path.join(dirroot,'images',imgname)
        imgpathnew = os.path.join(dirroot,'images_resize',imgname)
        maskpath = os.path.join(dirroot,'gt_image',imgname.replace('jpg','png'))

        if os.path.isfile(imgpath) and os.path.isfile(maskpath):
            img = cv2.resize(cv2.imread(imgpath),dsize= reshape)
            mask = cv2.resize(cv2.imread(maskpath),dsize= reshape)

            mask = mask[8:360,0:640,:]
            img = img[8:360,0:640,:]

            cv2.imwrite(imgpathnew,img)
            cv2.imwrite(maskpath,mask)

def get_txt():
    txt =[]
    filedir = '/home/wqg/data/tusimple/train'

    filtList = os.listdir(os.path.join(filedir,'images'))

    for name in filtList:
        imgpath = os.path.join('images',name)
        maskpath = os.path.join('gt_image',name.replace('jpg','png'))

        txt.append("{} {}\n".format(imgpath,maskpath))

    with open('/home/wqg/data/tusimple/train/train_gt.txt','w') as f:
        f.writelines(txt)

def get_se_txt():
    txt = []
    filedir = '/home/jovyan/data-vol-1/wqg/lane/'

    filelist = ['tusimple','CULane','CurveLanes']
    for file in filelist:
        if file == 'tusimple':
            filtList = os.listdir(os.path.join(filedir, file,'train/images'))
            for name in filtList:
                if os.path.splitext(name)[-1] != '.jpg':
                    continue
                imgpath = os.path.join(file,'train/images', name)
                maskpath = os.path.join(file,'train/gt_image', name.replace('jpg', 'png'))
                txt.append("{} {}\n".format(imgpath, maskpath))

        if file == 'CULane':
            filtList = os.listdir(os.path.join(filedir, file, 'images'))
            for name in filtList:
                if os.path.splitext(name)[-1] != '.jpg':
                    continue
                imgpath = os.path.join(file,'images', name)
                maskpath = os.path.join(file,'gt_image', name.replace('jpg', 'png'))
                txt.append("{} {}\n".format(imgpath, maskpath))

        if file == 'CurveLanes':
            filtList = os.listdir(os.path.join(filedir, file, 'train/images_resize'))
            for name in filtList:
                if os.path.splitext(name)[-1] != '.jpg':
                    continue
                imgpath = os.path.join(file, 'train/images_resize', name)
                maskpath = os.path.join(file, 'train/igt_image', name.replace('jpg', 'png'))
                txt.append("{} {}\n".format(imgpath, maskpath))
    with open(os.path.join(filedir,'train_gt.txt'),'w') as f:
        f.writelines(txt)


def get_lane_seg_precent():

    dit1 = {0: 464027126, 1: 114878306, 2: 6270039, 3: 318162336, 4: 140632979, 6: 45698798, 7: 16028045, 5: 161913284 ,'pic': 11350 }
    dit2 = {0: 464465060, 1: 114536258, 2: 6467828, 3: 318231193, 4: 140436528, 6: 46270648, 7: 16072567, 5: 161099696 ,'pic': 11350 }
    dit3 = {0: 463397315, 1: 114542088, 2: 6581968, 3: 317946512, 4: 140496933, 6: 46691664, 7: 16047623, 5: 159886437 ,'pic': 11350}
    dit4 = {0: 331045397, 1: 73861578, 2: 6579018, 3: 182493941, 4: 127233644, 6: 36778819, 7: 15912533, 5: 132035017  ,'pic': 11350}
    dit5 = {0: 461985720, 1: 115121502, 2: 6526393, 3: 317917468, 4: 140560308, 6: 45450486, 7: 16169982, 5: 158931421 ,'pic': 11350}
    dit6 = {0: 464269977, 1: 114845401, 2: 6345146, 3: 318170096, 4: 140609279, 6: 45747249, 7: 15940488, 5: 158586869 ,'pic': 11350}
    dit7 = {0: 464685261, 1: 114389310, 2: 6086347, 3: 318448204, 4: 140585910, 6: 45537773, 7: 15864074, 5: 158288043 ,'pic': 11350}
    dit8 = {0: 464588439, 1: 114711590, 2: 5926246, 3: 318313270, 4: 140353386, 6: 45203255, 7: 16102294, 5: 159789686 ,'pic': 11350}
    dit9 = {0: 465921212, 1: 114920596, 2: 5910965, 3: 318442841, 4: 140472175, 6: 45151800, 7: 16268279, 5: 161353113 ,'pic': 11350}
    dit10 = {0: 466754527, 1: 114922829, 2: 6093449, 3: 318424340, 4: 140668197, 6: 45190001, 7: 16169951, 5: 161658261 ,'pic': 11350}

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
    del dictsum[0]

    dictprecent = {}

    sums = sum(dictsum.values())
    for key in dictsum:
        name = namedict[key]
        dictprecent[name] = round(dictsum[key]/sums * 100, 4 )


    print(dictprecent)
"""
类别对应名称:
{0: '其他类别', 1: '白虚线', 2: '黄虚线', 3: '白实线', 4: '黄实线', 5: '导流线', 6: '停止线', 7: '禁停区', 8: '栏栅'}

图片数量:113653
{0: 4642937567, 1: 1147665713, 2: 62787399, 3: 3181918749, 4: 1405415110, 6: 456901282, 7: 160618047, 5: 1600968422, 'pic': 113653}
{'其他类别': 36.6764, '白虚线': 9.0659, '黄虚线': 0.496, '白实线': 25.1352, '黄实线': 11.1019, '停止线': 3.6092, '禁停区': 1.2688, '导流线': 12.6467}

不包含其他车道线各车道线占比
{'白虚线': 14.3167, '黄虚线': 0.7832, '白实线': 39.6932, '黄实线': 17.532, '停止线': 5.6997, '禁停区': 2.0036, '导流线': 19.9715}

"""

if __name__ == "__main__":
    # path = '/home/wqg/data/lane/'
    # # main(path)
    # dirroot = '/home/wqg/data/CurveLanes/train'
    # # resize_tusimple(dirroot)
    # # resize_CurveLanes(dirroot)
    # get_txt()

    get_lane_seg_precent()





