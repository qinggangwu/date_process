import random

import cv2
import os
from tqdm import tqdm

"""
针对于rec样本均衡进行处理
"""

def mkdir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def cat_img(strl):
    """
    裁剪对应缺少数据的图片(rec字符统计数量少于1000),存入对应的目录
    :return:
    """
    txts_dir ='/home/jovyan/data-vol-1/wqg/container/20220110_rec/valtest/labels'  # train valtest
    save_path = '/home/jovyan/data-vol-1/wqg/container/balanced/test'
    # strDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}
    labelDict= {}
    strDict = {}
    with open("contain_cha_region_det/classes.name") as f:
        class_name = [ss[:-1] for ss in f.readlines()]
        for idx, line in enumerate(class_name):
            strDict[line] = idx
            labelDict[idx] = line  # 存储序号对应的字符,对应取label

    labelList = [ strDict[p] for p in strl]
    txts = os.listdir(txts_dir)
    index = 0
    for i_txt in tqdm(range(len(txts))):
        txt = txts[i_txt]
        txt_path = os.path.join(txts_dir, txt)
        img_path = txt_path.replace('labels','images').replace('txt','jpg')
        lines = open(txt_path, 'r', encoding='utf-8').readlines()
        for line in lines:
            info = line.strip().split(' ')
            if int(info[0]) in labelList:
                label = labelDict[int(info[0]) ]

                point = info[1:]
                img = cv2.imread(img_path)
                h,w = img.shape[:2]    # img.shape -> h,w,c
                point1 = int(float(point[0])*w) ,int(float(point[1])*h)
                wh = int(float(point[2])*w/2),int(float(point[3])*h/2)

                imgcat = img[point1[1]- wh[1]: point1[1]+ wh[1],point1[0]-wh[0]: point1[0] +wh[0]]

                savePath = os.path.join(save_path,label)
                mkdir_dir(savePath)
                cv2.imwrite(os.path.join(savePath,'{}_{}.jpg'.format(label,index)),imgcat)
                index+=1

                # cv2.imshow(label,imgcat)
                # cv2.waitKey(1000)


def data_balanced(lackstr,balanstr):
    """

    :param lackstr: 缺少的字符
    :param balanstr: 较多的字符,将其替换成缺少的字符
    :return:
    """
    dirpath = '/home/jovyan/data-vol-1/wqg/container/20220110_rec/valtest/'  # train valtest
    catimgpath = '/home/jovyan/data-vol-1/wqg/container/balanced/test'
    # strDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}
    labelDict = {}
    strDict = {}
    with open("contain_cha_region_det/classes.name") as f:
        class_name = [ss[:-1] for ss in f.readlines()]
        for idx, line in enumerate(class_name):
            strDict[line] = idx
            labelDict[idx] = line  # 存储序号对应的字符,对应取label

    lackList = [strDict[p] for p in lackstr]
    balanList = [strDict[p] for p in balanstr]

    balantxtList = []
    txtPathList =[ os.path.join(dirpath,'labels',ll) for ll in os.listdir(os.path.join(dirpath,'labels')) if ll[-3:] == 'txt']

    for txtpath in txtPathList:
        lines = open(txtpath, 'r', encoding='utf-8').readlines()
        for line in lines:
            info = line.strip().split(' ')
            if int(info[0]) in balanList:
                balantxtList.append(txtpath)
                continue

    print(len(balantxtList))
    num = 0
    for s in lackList:
        label = labelDict[s]
        print(label)
        lackimgList = [ os.path.join(catimgpath, label,ll) for ll in os.listdir(os.path.join(catimgpath, label)) if ll[-3:] == 'jpg']
        for i in tqdm(range(1000)):
            txtpath = random.choice(balantxtList)
            imgpath = txtpath.replace('labels','images').replace('txt','jpg')
            lines = open(txtpath, 'r', encoding='utf-8').readlines()
            for lineindex,line in enumerate(lines):
                info = line.strip().split(' ')
                # print(info)
                if int(info[0]) in balanList:
                    point = info[1:]
                    img = cv2.imread(imgpath)
                    h, w = img.shape[:2]  # img.shape -> h,w,c
                    point1 = int(float(point[0]) * w), int(float(point[1]) * h)
                    wh =  int(float(point[2]) * w / 2),int(float(point[3]) * h / 2),

                    lackimgPath = random.choice(lackimgList)
                    lackimg = cv2.imread(lackimgPath)
                    # lackReszImg = cv2.resize(lackimg,(wh[1]*2 ,wh[0]*2  ))

                    try:
                        lackReszImg = cv2.resize(lackimg,(wh[0]*2 ,wh[1]*2  ))
                            #h -> w
                        img[point1[1]- wh[1]: point1[1]+ wh[1],point1[0]-wh[0]: point1[0] +wh[0]] =lackReszImg

                        #
                        # saveimgpath = imgpath.replace('container_rec/935_936','balanced')
                        # savetxtpath = txtpath.replace('container_rec/935_936','balanced')
                        #
                        # saveimgpath= saveimgpath[:-4] + "{}_{}.jpg".format('balance',num)
                        # savetxtpath= savetxtpath[:-4] + "{}_{}.txt".format('balance',num)
                        # num +=1
                        # print(info,s,info[0],type(s));quit()
                        info[0] = str(s)
                        reinfo = ' '.join(info) + " \n"
                        lines[lineindex] =reinfo

                        cv2.imwrite(imgpath,img)

                        with open(txtpath,'w',encoding='utf-8') as f:
                            f.writelines(lines)
                    except:
                        print(imgpath)
                    break
                    # cv2.imshow(label,img)
                    # cv2.waitKey(1000);quit()



                    # print(lackReszImg.shape)
                    # print(point1,wh)




if __name__ == "__main__":
    strs = 'JPQVWX'
#     strs = 'YZ'
    # strs = 'PWX'
    bastr = 'UG'  # 256+ 140 +11
    txts_dir = '/home/jovyan/data-vol-1/wqg/container/20220110_rec/train/labels'
    
#     cat_img(strs)
    data_balanced(strs,bastr)