import cv2
import os
from tqdm import tqdm
from xml.dom.minidom import parse
import xml.dom.minidom

class choice_Process():
    def __init__(self,dir_root,labelList):
        self.dir_root = dir_root
        self.labelList = labelList
        self.fps = 25
        self.num = 5    # 前后切取多少帧


    def video2img(self,video_path,numlist):
        """
        传入视频路径,输出image
        fps = 25
        :param video_path:
        :return:
        """

        fps = 25
        cap = cv2.VideoCapture(video_path)
        # filename = video_path.split("/")[-1][:-4]

        imgList = []
        relist = []
        n = 0
        # for i in range(760):
        #     ret, frame = cap.read()
        #     if ret == True:
        #         imgList.append(frame)
        #         cv2.imshow("video", frame)
        #         cv2.waitKey(10)
        # 判断载入的视频是否可以打开
        rval = cap.isOpened()
        while rval:  # 循环读取视频帧
            # index = index + 1
            rval, frame = cap.read()
            if rval:
                imgList.append(frame)
                cv2.imshow("video", frame)
                cv2.waitKey(10)



                # name = "{}_frame_{}.jpg".format(filename,n)
                # n+=1
                # savepath = os.path.join('/home/wqg/data/D2city/test',name)
                # cv2.imwrite(savepath,frame)
        cap.release()
        print(len(imgList))
        for ii in numlist:
            relist += imgList[ii:ii + 10]
        return relist


    def get_file_name(self,file):
        dir_root = os.path.join(self.dir_root,file)
        vidlist= [ii[:-4] for ii in os.listdir(dir_root) if ii[-3:] == 'mp4']
        labellist= [ii[:-4] for ii in os.listdir(dir_root) if ii[-3:] == 'xml']
        if len(vidlist) >= len(labellist):
            return labellist
        else:
            return vidlist

    def get_read_xml_bool(self,xmlpath):
        DOMTree = xml.dom.minidom.parse(xmlpath)
        collection = DOMTree.documentElement
        frameList = []
        messages = collection.getElementsByTagName("track")
        for message in messages:
            la = message.getAttribute('label')
            if la in self.labelList:
                frame = message.getElementsByTagName("box")[0].getAttribute('frame')
                frame = int(frame)
                if frame !=0:
                    frame = frame - self.num +1
                frameList.append(frame)
                # print(la,frame)
        if len(frameList)>0:
            return [True, list(set(frameList))]
        else:
            return [False]


def main():
    video_path = '/home/wqg/data/D2city/1001/0a5f04a6dcca47635c5176901e58753f.mp4'
    # video2img(video_path)
    dir_root = '/home/wqg/data/D2city/'
    labellist = ['van', 'bus', 'truck', 'bicycle', 'motorcycle', 'open-tricycle', 'closed-tricycle', 'forklift', 'large-block', 'small-block']
    labellist = ['truck', 'forklift']

    fileList = ['1001','1002','1003']
    fileList = ['1001']
    imgnum = 0

    P = choice_Process(dir_root,labellist)
    for fl in fileList:
        nameList = P.get_file_name('1001')

        for flname in nameList[:30]:
            xmlpath = os.path.join(dir_root,fl,flname+'.xml')
            flag = P.get_read_xml_bool(xmlpath)
            # print(flag)

            filenum = imgnum //500
            os.makedirs(os.path.join('/home/wqg/data/D2city/test',str(filenum)) ,exist_ok=True)

            if flag[0]:
                imgList = P.video2img(os.path.join(dir_root,fl,flname+'.mp4'),flag[1])
                for idx,img in enumerate(imgList):
                    n = flag[1][idx//10]
                    m = idx%10
                    name = "{}_frame_{}.jpg".format(flname,n+m)
                    savepath = os.path.join('/home/wqg/data/D2city/test',str(filenum),name)
                    cv2.imwrite(savepath,img)
                    imgnum +=1

                    # print(savepath)
                # print(len(imgList))
                # print(flag);quit()


if __name__=="__main__":
    main()
    # s = 'car,van,bus,truck,person,bicycle,motorcycle,open-tricycle,closed-tricycle,forklift,large-block,small-block'
    # print(s.split(','))
