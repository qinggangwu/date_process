# -*- coding: utf-8 -*-

"""
对像素进行聚类。
在像素级水平进行聚类可以用在一些很简单的图像

载入图像，并将其下采样到一个较低的分辨率，然后对这些区域用k-means进行聚类

K-means 的输入是一个有 stepsX × stepsY 行的数组,数组的每一行有 3 列,各列分别为区域块 R、G、B 三个通道的像素平均值。

为可视化最后的结果 , 我们用 SciPy 的imresize() 函数在原图像坐标中显示这幅图像。

参数 interp 指定插值方法;我们在这里采用最近邻插值法,以便在类间进行变换时不需要引入新的像素值。

"""
import cv2
from scipy.cluster.vq import *
# from scipy.misc import imresize
from skimage.transform import resize as imresize
from pylab import *

from PIL import Image

#steps*steps像素聚类
def clusterpixels_square(infile, k, steps):

    im = array(Image.open(infile))

    #im.shape[0] 高 im.shape[1] 宽
    dx = im.shape[0] // steps
    dy = im.shape[1] // steps
    # 计算每个区域的颜色特征
    features = []
    for x in range(steps):
        for y in range(steps):
            R = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
            G = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R])
    features = array(features, 'f')     # 变为数组
    # 聚类， k是聚类数目
    centroids, variance = kmeans(features, k)
    code, distance = vq(features, centroids)

    # 用聚类标记创建图像
    codeim = code.reshape(steps, steps)

    # codeim = imresize(codeim, im.shape[:2], 'nearest')
    # codeim = codeim.resize(im.shape[:2], Image.ANTIALIAS)
    return codeim

#stepsX*stepsY像素聚类
def clusterpixels_rectangular(infile, k, stepsX):

    im = array(Image.open(infile))

    stepsY = stepsX * im.shape[1] // im.shape[0]

    #im.shape[0] 高 im.shape[1] 宽
    dx = im.shape[0] // stepsX
    dy = im.shape[1] // stepsY
    # 计算每个区域的颜色特征
    features = []

    for x in range(int(stepsX)):
        for y in range(int(stepsY)):
            # R = im[: ,: ,0]
            R = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
            G = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R])


    # features = array(im[: ,: ,0], 'f')     # 变为数组
    features = array(features, 'f')     # 变为数组
    # 聚类， k是聚类数目
    centroids, variance = kmeans(features, k)
    # for
    s = centroids[:5,:]

    # code, distance = vq(features, centroids)
    code, distance = vq(features, s)
      # 用聚类标记创建图像
    codeim = code.reshape(int(stepsX), int(stepsY))

    # cv2.imshow('codeim',codeim)
    # cv2.waitKey(5000)

    # codeim = imresize(codeim, im.shape[:2], 'nearest')
    # print(im.shape[:2])
    # codeim = codeim.resize((im.shape[1],im.shape[0]), Image.ANTIALIAS)
    return codeim



#计算最优steps 为保证速度以及减少噪点 最大值为maxsteps 其值为最接近且小于maxsteps 的x边长的约数
def getfirststeps(img,maxsteps):
    msteps = img.shape[0]
    n = 2
    while(msteps>maxsteps):
        msteps = img.shape[0]//n
        n = n + 1
    return msteps

#Test


#图像文件 路径
# infile = './data/10.jpg'
infile = '/home/wqg/pyproject/git/lane/lanenet-lane-detection-pytorch/logwu/instance_output.jpg'

im = array(Image.open(infile))

#参数
m_k = 10

m_maxsteps = 128

#显示原图empire.jpg
figure()

subplot(121)
title('source')
imshow(im)


# 用改良矩形块对图片的像素进行聚类
codeim= clusterpixels_rectangular(infile, m_k,getfirststeps(im,m_maxsteps))

subplot(122)
title('New steps = '+str(getfirststeps(im,m_maxsteps))+' K = '+str(m_k));
imshow(codeim)

# #方形块对图片的像素进行聚类
# codeim= clusterpixels_square(infile, 15, 200)
# subplot(133)
# title('Old steps = 200 K = '+str(m_k));

# imshow(codeim)

show()

