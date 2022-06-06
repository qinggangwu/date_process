


import copy
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
import time
import matplotlib.pyplot as plt
from scipy import optimize



class fit_lane:
    def __init__(self ,imgnpy,masknpy,shape):
        self.imgnpy = imgnpy
        self.masknpy = masknpy
        self.imgshape = shape     # (c,h,w)
        self.eps = 1.5                 # 需要聚类的车道线聚类半径    3
        self.MinPts = 2              # 需要聚类的车道线聚类点个数   5
        self.ypoint = 95           # 小于96的值直接舍弃    95第8张图片的临界值

    # 透视变换
    def warpImage(self,image, M):
        image_size = (self.imgshape[2], self.imgshape[1])
        # rows = img.shape[0] 256
        # cols = img.shape[1] 512
        # M = cv2.getPerspectiveTransform(src_points, dst_points)
        # Minv = cv2.getPerspectiveTransform(dst_points, src_points)
        warped_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
        return warped_image

    def img_DBSCAN(self,img):

        """

        :param img: 输入的是 nask图片, np.load()读取结果
        :return:
        """

        pointTuple = np.where(img == 1)                 # 获取坐标点        返回tuple (y:array ,x:array)
        whereflag = pointTuple[0] > self.ypoint
        pointlist = np.array([pointTuple[1],pointTuple[0]]).swapaxes(0,1)[whereflag]   # 获取所有满足的点坐标
        pointdict = {}

        # img0 = cv2.imread("/home/wqg/pyproject/git/lane/lanenet-lane-detection-pytorch/logwu/input17.jpg")
        # for xx in pointlist:
        #     cv2.circle(img0, xx, 1, (0, 0, 255), 4)
        # cv2.imshow("mask",img0)
        # cv2.waitKey(0)

        print( "密度聚类点的数量为: ",len(pointlist))
        # db = DBSCAN(eps=0.35, min_samples=1000)

        # 密度聚类
        # eps = self.eps # 领域的大小，使用圆的半径表示
        # MinPts = self.MinPts  # 领域内，点个数的阈值
        db = DBSCAN(eps=self.eps, min_samples=self.MinPts)
        model0 = db.fit(pointlist)
        labels = model0.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
        # print(n_clusters_)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)  # 设置一个样本个数长度的全false向量
        core_samples_mask[db.core_sample_indices_] = True  # 将核心样本部分设置为true
        for i in range(n_clusters_):
            class_member_mask = (labels == i)  # 将所有属于该聚类的样本位置置为true
            pointdict[i] = pointlist[class_member_mask & core_samples_mask]  # 将所有属于该类的核心样本取出，使用大图标绘制  np.array [(x,y)]


        return pointdict


        # 聚类过程可视化
        import seaborn as sns
        X = pointlist
        sns.set(font='SimHei', style='ticks')
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        fig = plt.figure(figsize=(12, 5))

        ax = fig.add_subplot(1, 2, 1)
        row, _ = np.shape(X)
        # 画子图，未聚类点
        for i in range(row):
            ax.plot(X[i, 0], X[i, 1], '#4EACC5', marker='.')
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)  # 设置一个样本个数长度的全false向量
        core_samples_mask[db.core_sample_indices_] = True  # 将核心样本部分设置为true

        ax = fig.add_subplot(1, 2, 2)
        for k, col in zip(unique_labels, colors):
            if k == -1:  # 聚类结果为-1的样本为离散点
                # 使用黑色绘制离散点
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)  # 将所有属于该聚类的样本位置置为true
            xy = X[class_member_mask & core_samples_mask]  # 将所有属于该类的核心样本取出，使用大图标绘制
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=4)
            xy = X[class_member_mask & ~core_samples_mask]  # 将所有属于该类的非核心样本取出，使用小图标绘制
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=1)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        sns.despine()
        plt.show()
        return pointdict
        #

    def get_lane_mask(self,pointList):

        pointList = np.array(pointList).swapaxes(1,0)
        mask = np.zeros(self.imgshape[1:], np.uint8)
        mask[pointList[0],pointList[1]] = 255

        ## 图像的腐蚀,膨胀.默认迭代次数
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(mask, kernel)
        dst = cv2.dilate(erosion, kernel)

        pointTuple = np.where(dst > 5)

        if len(pointTuple[0]) <20:
            return np.array([])
        else:
            return pointTuple

    def get_point(self,pointlist,M):
        repoint =[]
        ypoint = pointlist[:,1]
        xpoint = pointlist[:,0]
        ymin ,ymax = ypoint.min(),ypoint.max()

        stap = 1
        # repoint = [[xpoint[ypoint == (y+ stap//2 +3) ].mean(), y] for y in range(ymin,ymax,stap)]

        for y in range(ymin,ymax,stap ):
            w1 = np.where(ypoint == y + stap//2)
            if len(w1[0]) == 0:
                continue
            else:
                # x1 = xpoint[w1].sum()/len(w1[0])
                repoint.append([xpoint[w1].mean(), y + stap//2])

        # print();quit()


        zj = np.array([1] * len(repoint)).reshape(-1, 1)
        point = np.concatenate((np.array(repoint), zj), axis=1)  # .swapaxes(0,1)
        Mpoint = np.dot(point, M.T)
        # Mpoint = np.dot(point, M)
        ss = np.vstack([Mpoint[:, 2], Mpoint[:, 2], Mpoint[:, 2]]).swapaxes(1, 0)
        Mpoint = np.true_divide(Mpoint, ss)

        return Mpoint

    def fit_line(self,Mpoint):
        """
        # 方程拟合
        :param Mpoint:
        :return:
        """

        z1 = np.polyfit(Mpoint[:,1], Mpoint[:,0], 3)  # 用3次多项式拟合，输出系数从高到0
        # return z1
        p1 = np.poly1d(z1)        # 使用次数合成多项式
        y_pre = p1(Mpoint[:,1])


        print('z1',z1)

        plt.plot(Mpoint[:,1], Mpoint[:,0], '.')
        plt.plot(Mpoint[:,1], y_pre)
        plt.show()
        # time.sleep(10)

        # def f_1(x, A, B, C, D):
        #     return A * x ** 3 + B * x **2 + C *x +D
        #
        # x_group = list(Mpoint[:,0])
        # y_group = list(Mpoint[:,1])
        #
        # # 得到返回的各个参数值值
        # A, B, C, D = optimize.curve_fit(f_1, x_group, y_group)[0]
        # # popt, pcov= optimize.curve_fit(func, x_group, y_group)
        #
        # # 数据点与原先的进行画图比较
        # plt.scatter(x_group, y_group, marker='o', label='真实值')
        # # x = np.arange(Mpoint[:,0].min(), Mpoint[:,0].max() , 0.1)
        # x = np.arange(Mpoint[:,0].min(), Mpoint[:,0].max())
        # y = A * x**3 + B * x**2 + C*x + D
        # plt.plot(x, y, color='red', label='拟合曲线')
        # plt.legend()  # 显示label
        #
        # plt.show()

def main():
    id = 8
    filename = 'logwu'
    imgpath = "/home/wqg/pyproject/git/lane/lanenet-lane-detection-pytorch/{}/input{}.jpg".format(filename,id)
    makspath = "/home/wqg/pyproject/git/lane/lanenet-lane-detection-pytorch/{}/binary_output{}.jpg".format(filename,id)
    imgnpy = "/home/wqg/pyproject/git/lane/lanenet-lane-detection-pytorch/{}/instance_output{}.npy".format(filename,id)
    masknpy = "/home/wqg/pyproject/git/lane/lanenet-lane-detection-pytorch/{}/binary_output{}.npy".format(filename,id)
    shape = (3, 256, 512)

    # mask = cv2.imread(makspath)
    # cv2.line(mask, (0, 90), (512, 90), (255, 0, 0), 1)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # print();quit()

    t = time.time()
    P = fit_lane(imgnpy, masknpy, shape)

    M = np.array([[-5.10096576e-01, -3.36259877e+00, 3.94451273e+02],
                  [2.22044605e-16, -3.59613696e+00, 3.23652327e+02],
                  [4.49142058e-19, -1.30151174e-02, 1.00000000e+00], ])
    Minv = np.array([[3.35937500e-01, -9.65576172e-01, 1.80000000e+02],
                     [0.00000000e+00, -2.78076172e-01, 9.00000000e+01],
                     [-0.00000000e+00, -3.61919403e-03, 1.00000000e+00]])

    img = cv2.imread(imgpath)
    imgwarp = P.warpImage(img, M)

    t0 = time.time()

    pointDict = P.img_DBSCAN(np.load(P.masknpy))
    print("聚类时间 : " , time.time() - t0)

    t1 = time.time()
    # for k,val in pointDict.items():
    #     print( k , len(val))


    pointlist = sorted(pointDict.items(), key=lambda x: len(x[1]), reverse=True)   # 对字典进行排序,得到列表,取列表的前4个,就是获取最长的4条线.

    for k,point in pointlist[:4]:
        if len(point) <150:
            continue
        point = P.get_point(point,M)

    # print("投影点转化时间 : ", time.time() - t1)

        # 绘制点到透视变换后的图像上
        # for xy in point:
        #     cv2.circle(imgwarp, [int(xy[0]),int(xy[1])], 1, (0, 0, 255), 4)
        Z = np.polyfit(point[:, 1], point[:, 0], 2)   # 曲线拟合,用y 拟合x

        print(Z)
        p1 = np.poly1d(Z)
        # print(p1)
        Y = np.array([i*3 for i in range(86)])
        X = p1(Y)

        for ii in range(85):
            if 0 <= X[ii] <= 512 and 0 <= X[ii+1] <= 512:
                # print((X[ii], Y[ii]), (X[ii+1], Y[ii+1]))
                cv2.line(imgwarp, (int(X[ii]), Y[ii]), (int(X[ii+1]), Y[ii+1]), (0, 0, 255), 1)

        cv2.imshow('maskwarpPoint', imgwarp)
    # cv2.imshow('img', img)
    #     cv2.waitKey(0)


    print(time.time() -t);quit()




if __name__ == "__main__":
    main()


"""
密度聚类点的数量为:  5550
聚类时间 :  0.020898818969726562
投影点转化时间 :  0.028650283813476562


密度聚类点的数量为:  4214
聚类时间 :  0.015763044357299805
投影点转化时间 :  0.025648117065429688


"""