
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
import time
import matplotlib.pyplot as plt
from scipy import optimize

# 透视变换
def warpImage(image, M):
    imgshape = image.shape
    # image_size = (self.imgshape[2], self.imgshape[1])
    image_size = (imgshape[1], imgshape[0])
    # rows = img.shape[0] ->h
    # cols = img.shape[1] ->w
    # M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    return warped_image

def get_wam_point(x,y,M):

    repoint =[x,y]
    # ypoint = x
    # xpoint = y
    # ymin ,ymax = ypoint.min(),ypoint.max()
    #
    # stap = 1
    # # repoint = [[xpoint[ypoint == (y+ stap//2 +3) ].mean(), y] for y in range(ymin,ymax,stap)]
    #
    # for y in range(ymin,ymax,stap ):
    #     w1 = np.where(ypoint == y + stap//2)
    #     if len(w1[0]) == 0:
    #         continue
    #     else:
    #         # x1 = xpoint[w1].sum()/len(w1[0])
    #         repoint.append([xpoint[w1].mean(), y + stap//2])

    # print();quit()

    zj = np.array([1] * len(repoint[0])).reshape(1, -1)
    point = np.concatenate((np.array(repoint), zj), axis=0).swapaxes(0,1)  # .swapaxes(0,1)
    Mpoint = np.dot(point, M.T)
    # Mpoint = np.dot(point, M)
    ss = np.vstack([Mpoint[:, 2], Mpoint[:, 2], Mpoint[:, 2]]).swapaxes(1, 0)
    Mpoint = np.true_divide(Mpoint, ss)

    return Mpoint

def drow_tusimle(M,Minv):
    filename = '01895'  # 弯道图片
    # filename = '03725'
    filepath = 'driver_37_30frame/05191626_0491.MP4'

    imgflie = '/home/wqg/pyproject/git/lane/tusimple_test_image'
    txtname = '/home/wqg/pyproject/git/lane/Ultra-Fast-Lane-Detection/tmp/tusimple_eval_tmp.0.txt'

    flag = False


    txtinfo = open(txtname, 'r').readlines()

    for flie in txtinfo:
        flie = eval(flie)
        imgname = flie['raw_file']

        img = cv2.imread(os.path.join(imgflie,imgname))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        lineinfo = flie['lanes']
        lineY = flie['h_samples']

        for line in lineinfo:
            X = []
            Y = []
            for ind,x in enumerate(line):
                if x == -2 or ind >len(lineY)-1:
                    continue
                # print(ind,x,lineY[ind])
                cv2.circle(img, [x,int(lineY[ind])], 1, (0, 0, 255), 4)
                X.append(x)
                Y.append(int(lineY[ind]))

            Z = np.polyfit(Y, X, 5)  # 曲线拟合,用y 拟合x
            print(Z)
            p1 = np.poly1d(Z)
            # print(p1)
            YY = np.array([i * 5 for i in range(40, 144)])
            XX = p1(YY)

            for ii in range(143-40):
                if 0 <= XX[ii] <= 1280 and 0 <= XX[ii + 1] <= 1280:
                    try:
                        cv2.line(img, (int(XX[ii]), YY[ii]), (int(XX[ii + 1]), YY[ii + 1]), (0, 0, 255), 1)
                    except:
                        print((XX[ii], YY[ii]), (XX[ii + 1], YY[ii + 1]))
            cv2.imshow('img', img)
            cv2.waitKey(0)


        if flag:
            img = cv2.resize(img,(512,256))
            warpimg = warpImage(img, M)
            for line in lineinfo:
                X = []
                Y = []
                for ind,x in enumerate(line):
                    if x == -2:
                        continue
                    print(int(x*512/1280 ),int(lineY[ind]*256/720))
                    cv2.circle(img, [int(x*512/1280 ),int(lineY[ind]*256/720)], 1, (0, 0, 255), 4)
                    X.append(x*512/1280)
                    Y.append(lineY[ind]*256/720)
                cv2.imshow('img', img)
                cv2.waitKey(0)

                point = get_wam_point(X,Y,M)
                Z = np.polyfit(point[:, 1], point[:, 0], 2)  # 曲线拟合,用y 拟合x


                print(Z)
                p1 = np.poly1d(Z)
                # print(p1)
                YY = np.array([i * 3 for i in range(0, 85)])
                XX = p1(YY)

                for ii in range(84):
                    if 0 <= XX[ii] <= 512 and 0 <= XX[ii + 1] <= 512:

                        try:
                            cv2.line(warpimg, (int(XX[ii]), YY[ii]), (int(XX[ii + 1]), YY[ii + 1]), (0, 0, 255), 1)
                        except:
                            print((XX[ii], YY[ii]), (XX[ii + 1], YY[ii + 1]))

                cv2.imshow('img', warpimg)
                cv2.waitKey(0)
        # print(txtinfo)


def drow_culane(M,Minv):
    filename = '01895'  # 弯道图片
    # filename = '03725'
    filepath = 'driver_37_30frame/05191626_0491.MP4'

    imgname = '/home/wqg/data/CULane/{}/{}.jpg'.format(filepath,filename)
    txtname = '/home/wqg/pyproject/git/lane/Ultra-Fast-Lane-Detection/tmp/culane_eval_tmp/{}/{}.lines.txt'.format(filepath,filename)

    imgname = '/home/wqg/pyproject/gitLib/lane_det/20220225/samples/CULane_test_image/03360.jpg'
    txtname = '/home/wqg/pyproject/gitLib/lane_det/20220225/samples/result/culane_eval_tmp/03360.lines.txt'

    img = cv2.imread(imgname)
    txtinfo = open(txtname,'r').readlines()

    for line in txtinfo:
        line =line.split()
        for i in range(len(line)//2):
            x = line[2*i]
            y = line[2*i+1]
            cv2.circle(img, [int(x),int(y)], 1, (0, 0, 255), 4)


        X = np.array([int(line[2*x]) for x in range(len(line)//2)])
        Y = np.array([int(line[2*x+1]) for x in range(len(line)//2)])

        Z = np.polyfit(Y, X, 3)  # 曲线拟合,用y 拟合x

        print(Z)
        p1 = np.poly1d(Z)
        # print(p1)
        YY = np.array([i * 3 for i in range(80,195)])
        XX = p1(YY)

        for ii in range(194-80):
            if 0 <= XX[ii] <= 1640 and 0 <= XX[ii + 1] <= 1640:

                try:
                    cv2.line(img, (int(XX[ii]), YY[ii]), (int(XX[ii + 1]), YY[ii + 1]), (0, 0, 255), 1)
                except:
                    print((XX[ii], YY[ii]), (XX[ii+1], YY[ii+1]))

        cv2.imshow('img',img)
        cv2.waitKey(0)
    print(txtinfo)



if __name__ =="__main__":
    M = np.array([[-5.10096576e-01, -3.36259877e+00, 3.94451273e+02],
                  [2.22044605e-16, -3.59613696e+00, 3.23652327e+02],
                  [4.49142058e-19, -1.30151174e-02, 1.00000000e+00], ])
    Minv = np.array([[3.35937500e-01, -9.65576172e-01, 1.80000000e+02],
                     [0.00000000e+00, -2.78076172e-01, 9.00000000e+01],
                     [-0.00000000e+00, -3.61919403e-03, 1.00000000e+00]])
    drow_culane(M,Minv)
    # drow_tusimle(M,Minv)