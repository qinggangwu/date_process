import copy
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
import time
import matplotlib.pyplot as plt
from scipy import optimize

class fit_lane:
    def __init__(self ,imgpath,maskpath,shape):
        self.imgpath = imgpath
        self.maskpath = maskpath
        self.imgshape = shape     # (c,h,w)
        self.m = 6              # 需要聚类的车道线条数

    def warpImage(self,image, M):
        image_size = (self.imgshape[2], self.imgshape[1])
        # rows = img.shape[0] 256
        # cols = img.shape[1] 512

        # M = cv2.getPerspectiveTransform(src_points, dst_points)
        # Minv = cv2.getPerspectiveTransform(dst_points, src_points)
        warped_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

        return warped_image

    def img_Preview(self):
        """
        :param img:
        :param mask:
        :return:
        """

        img = np.load(self.imgpath)
        mask = np.load(self.maskpath)

        def replace_color(image, color_codes, dst_clr):
            img_arr = np.asarray(image, dtype=np.double)
            r_img = img_arr[0, :, :].copy()
            g_img = img_arr[1, :, :].copy()
            b_img = img_arr[2, :, :].copy()
            img = r_img * 256 + g_img * 256 + b_img * 256
            src_color = color_codes[0] * 256 + color_codes[1] * 256 + color_codes[2] * 256  # 编码

            r_img[img < src_color] = dst_clr[0]
            g_img[img < src_color] = dst_clr[1]
            b_img[img < src_color] = dst_clr[2]

            dst_img = np.array([r_img, g_img, b_img], dtype=np.float32)
            dst_img = dst_img

            return dst_img

        img = replace_color(img, [0.02,0.02,0.02],[0.04, 0.04, 0.04])
        # img = replace_color(img, [0.02,0.02,0.02],[0.94, 0.94, 0.94])

        imgre = np.multiply(img,mask)
        # img = cv2.bitwise_or(img,mask)

        # imgre = imgre.transpose((1, 2, 0))
        # cv2.imshow('img',imgre.transpose((1, 2, 0)))
        # cv2.waitKey(0)
        return imgre

    def img_Kmean(self,img):

        imgone = img[0,:,:] + img[1,:,:] + img[2,:,:]
        s = 0.04
        pointTuple = np.where(imgone > s)                 # 获取坐标点        返回tuple (x:array ,y:array)
        color = img[:,pointTuple[0],pointTuple[1]]        # 像素值

        all_pointlist = np.array(pointTuple).swapaxes(0,1)
        all_color = color.swapaxes(1,0)
        # print(color)
        print('需要聚类点个数', len(all_pointlist))



        # t0 = time.time()
        # m = 6   # 设置初始聚类中心个数
        kmeans = KMeans(n_clusters=self.m)
        kmeans.fit(all_color)
        color_label = {}
        for i in range(len(kmeans.cluster_centers_)):
            color_label[i] = kmeans.cluster_centers_[i]

        # for key,val in color_label.items():
        #     print("color :{}  val: {}".format(key,val))
        # print(color_label)


        # 去除多余聚类中心
        color_list = [color_label[color] for color in color_label]
        num = []
        for x in range(len(color_list)):
            for y in range(1+x,len(color_list) ):
                X = color_list[x]
                Y = color_list[y]
                a = np.linalg.norm(X - Y)
                if a < 0.5:
                    num.append(y)
        num = list(set(num))
        num.sort(reverse=True)
        for inx in num:
            color_list.pop(inx)


        assert len(all_color) ==len(all_pointlist)

        # 计算点的归属那个lane
        fit_point = {str(color.tolist()):[] for color in color_list }
        for ind,point in enumerate(all_pointlist):
            for color in color_list:
                if np.linalg.norm(all_color[ind] - color) < 0.5:
                    fit_point[str(color.tolist())].append(point)
                    continue
        return fit_point
        # for key ,val in fit_point.items():
        #     print(key)
        #     print(len(val))

        # m = m-len(num)
        # print('计算后聚类中心个数',m)

    def get_lane_mask(self,color,pointList):

        pointList = np.array(pointList).swapaxes(1,0)
        mask = np.zeros(self.imgshape[1:], np.uint8)
        mask[pointList[0],pointList[1]] = 255

        ## c.图像的腐蚀,膨胀.默认迭代次数
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(mask, kernel)
        dst = cv2.dilate(erosion, kernel)

        pointTuple = np.where(dst > 5)

        if len(pointTuple[0]) <20:
            return np.array([])
        else:
            return pointTuple


    def get_point(self,pointtuple,M):

        # 处理全部点
        # zj = np.array([1]* len(pointtuple[0])).reshape(1,-1)
        # pointre = np.array([pointtuple[1],pointtuple[0]])   # 换成x,y 形式
        # point = np.concatenate((pointre, zj), axis=0).swapaxes(0,1)
        # Mpoint = np.dot(point, M.T)
        # ss = np.vstack([Mpoint[:, 2], Mpoint[:, 2], Mpoint[:, 2]]).swapaxes(1, 0)
        # Mpoint = np.true_divide(Mpoint, ss)

        repoint =[]
        ypoint = pointtuple[0]
        xpoint = pointtuple[1]
        ymin ,ymax = ypoint.min(),ypoint.max()
        # if ymin < 100:
        #     ymin =100
        for y in range(ymin,ymax ):
            # w0 = np.where(ypoint == y)
            w1 = np.where(ypoint == y)
            # w2 = np.where(ypoint == y+2)
            # if len(w0[0])==0 or len(w1[0])==0 or len(w2[0])==0 :
            if len(w1[0])==0 :
                continue
            else:
                # x0 = xpoint[w0].sum()/len(w0[0])
                x1 = xpoint[w1].sum()/len(w1[0])
                # x2 = xpoint[w2].sum()/len(w2[0])
                repoint.append([x1, y])

        zj = np.array([1] * len(repoint)).reshape(-1, 1)
        point = np.concatenate((np.array(repoint), zj), axis=1)  # .swapaxes(0,1)
        Mpoint = np.dot(point, M.T)
        ss = np.vstack([Mpoint[:, 2], Mpoint[:, 2], Mpoint[:, 2]]).swapaxes(1, 0)
        Mpoint = np.true_divide(Mpoint, ss)

        return Mpoint

    # # 密度聚类
    # eps = 0.00005 # 领域的大小，使用圆的半径表示
    # MinPts = 1.5  # 领域内，点个数的阈值
    # model = DBSCAN(eps=eps, min_samples=MinPts)
    # model0 = model.fit(np.array(all_pointlist))
    # labels = model0.labels_
    # print(labels)
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    # print(n_clusters_)
    #
    # # # 数据匹配
    # all_point['type'] = model.fit_predict(all_pointlist)
    # # # 绘图
    # plt.scatter(
    #     all_point['x'],
    #     all_point['y'],
    #     c=all_point['type'] ) # 表示颜色
    # plt.show()





    # color_num = {}
    # for m in range(len(np.unique(kmeans.labels_))):
    #     # print('sum',np.sum(kmeans.labels_ == m))
    #     print(np.sum(kmeans.labels_ == m) ,color_label[m])  # 标签m对应的色彩
    #     color_num[np.sum(kmeans.labels_ == m)] = color_label[m]
    # print('color_num',color_num)

def main():
    id = 19
    imgpath = "/home/wqg/pyproject/git/lane/lanenet-lane-detection-pytorch/{}/input{}.jpg".format("tu", id)
    makspath = "/home/wqg/pyproject/git/lane/lanenet-lane-detection-pytorch/{}/binary_output{}.jpg".format("tu", id)
    imgnpy = "/home/wqg/pyproject/git/lane/lanenet-lane-detection-pytorch/{}/instance_output{}.npy".format("tu", id)
    masknpy = "/home/wqg/pyproject/git/lane/lanenet-lane-detection-pytorch/{}/binary_output{}.npy".format("tu", id)

    shape = (3, 256, 512)

    t = time.time()
    P = fit_lane(imgnpy, masknpy, shape)

    # 计算M矩阵
    # src_points =np.float32([[0, 256],[180, 90], [352, 90], [512, 256]])
    # dst_points = np.float32([[200, 256],[0, 0], [512, 0], [312, 256]])
    # M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    # print(M)
    # print(Minv);quit()

    M = np.array([[-5.10096576e-01 ,-3.36259877e+00,  3.94451273e+02],
                 [2.22044605e-16, -3.59613696e+00,  3.23652327e+02],
                 [4.49142058e-19, -1.30151174e-02,  1.00000000e+00],])
    Minv =np.array( [[ 3.35937500e-01 ,-9.65576172e-01 , 1.80000000e+02],
                     [ 0.00000000e+00 ,-2.78076172e-01 , 9.00000000e+01],
                     [-0.00000000e+00 ,-3.61919403e-03 , 1.00000000e+00]])

    # point = np.float32([[80,256,1],[90,256,1],[100,256,1],[110,256,1],[120,256,1]])
    #
    # point_inv=np.float32( [[217.50000059 ,256.00000036  , 1.],
    #          [219.68750059 ,256.00000036  , 1.],
    #         [221.8750006,    256.00000036,    1.],
    #         [224.06250061, 256.00000036 ,  1.],
    #         [226.25000061,    256.00000036,    1.]])

    # Mpoint= cv2.perspectiveTransform(point,M)
    #
    # print(point.shape)
    # Mpoint = np.dot(point,M)
    #
    # for ind,ii in enumerate(Mpoint):
    #     # print(i,i[2])
    #      Mpoint[ind] = np.true_divide(ii,np.array(ii[2]))
    #      # Mpoint[ind] = np.dot(ii,np.array(ii[2]))
    # print(Mpoint);quit()

    img = cv2.imread(imgpath)
    imgwarp =P.warpImage(img,M)
    #
    mask = cv2.imread(makspath)
    mask0 =P.warpImage(mask,M)

    reimg = P.img_Preview()
    pointDict = P.img_Kmean(reimg)


    for color,pointList in pointDict.items():
        pointre=P.get_lane_mask(color,pointList)
        if len(pointre) == 0:
            continue
        Mpoint = P.get_point(pointre,M)


        point = Mpoint
        Z = np.polyfit(point[:, 1], point[:, 0], 2)  # 曲线拟合,用y 拟合x
        print(Z)
        p1 = np.poly1d(Z)
        Y = np.array([i * 3 for i in range(86)])
        X = p1(Y)

        for ii in range(85):
            if 0 <= X[ii] <= 512 and 0 <= X[ii + 1] <= 512:
                # print((X[ii], Y[ii]), (X[ii+1], Y[ii+1]))
                cv2.line(imgwarp, (int(X[ii]), Y[ii]), (int(X[ii + 1]), Y[ii + 1]), (0, 0, 255), 1)

        cv2.imshow('maskwarpPoint', imgwarp)
        # cv2.imshow('img', img)
        cv2.waitKey(0)


        # 透视变换的图像上瞄点
        # for xx in Mpoint:
        #     try:
        #         cv2.circle(img, [int(xx[0]),int(xx[1])], 1, (0, 0, 255), 4)
        #     except:
        #         print((xx[0]),(xx[1]))

        # for xx in point:
        #     cv2.circle(mask, xx, 1, (0, 0, 255), 4)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(1000)
        # continue

        # plt.plot([xx[1] for xx in point ], [xx[0] for xx in point ], '.')
        # plt.show()




        # print(pointre.shape)


        #
        # zj = np.array([1]* len(point)).reshape(-1,1)
        # point = np.concatenate((np.array(point),zj),axis=1) #.swapaxes(0,1)
        # Mpoint = np.dot(point, M.T)
        # ss = np.vstack([Mpoint[:,2],Mpoint[:,2],Mpoint[:,2]]).swapaxes(1,0)
        # Mpoint = np.true_divide(Mpoint, ss)


        # s0 = Mpoint[:,0] < 1000  # or Mpoint[:,1] < 1000
        # Mpoint = Mpoint[s0]
        #
        # s0 = Mpoint[:, 1] < 1000  # or Mpoint[:,1] < 1000
        # Mpoint = Mpoint[s0]
        #
        # s1 = Mpoint[:, 0] > 0
        # Mpoint = Mpoint[s1]
        #
        # s1 = Mpoint[:, 1] > 0
        # Mpoint = Mpoint[s1]



        # cv2.imshow('mask', img)
        # cv2.waitKey(5000)

        # 方程拟合

        # z1 = np.polyfit(Mpoint[:,0], Mpoint[:,1], 3)  # 用3次多项式拟合，输出系数从高到0
        # p1 = np.poly1d(z1)  # 使用次数合成多项式
        # y_pre = p1(Mpoint[:,0])
        # print('z1',z1)
        #
        # plt.plot(Mpoint[:,0], Mpoint[:,1], '.')
        # plt.plot(Mpoint[:,0], y_pre)
        # plt.show()
        # time.sleep(10)

        # print(len(pointre))
        #

    #     if len(Mpoint) < 4:
    #         continue
    #     def f_1(x, A, B, C, D):
    #         return A * x ** 3 + B * x **2 + C *x +D
    #
    #     x_group = list(Mpoint[:,0])
    #     y_group = list(Mpoint[:,1])
    #
    #     # 得到返回的各个参数值值
    #     A, B, C, D = optimize.curve_fit(f_1, x_group, y_group)[0]
    #     # popt, pcov= optimize.curve_fit(func, x_group, y_group)
    #
    #     # 数据点与原先的进行画图比较
    #     plt.scatter(x_group, y_group, marker='o', label='真实值')
    #     # x = np.arange(Mpoint[:,0].min(), Mpoint[:,0].max() , 0.1)
    #     x = np.arange(Mpoint[:,0].min(), Mpoint[:,0].max())
    #     y = A * x**3 + B * x**2 + C*x + D
    #     plt.plot(x, y, color='red', label='拟合曲线')
    #     plt.legend()  # 显示label
    #
    #     plt.show()
    #
    # cv2.imshow('image', img)
    # # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    print("点聚类时间",time.time() - t)


if __name__ == "__main__":
    main()








