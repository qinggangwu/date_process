import json
import os
import cv2
from tqdm import tqdm
import shutil


class porcess:
    def __init__(self,root ,fileList):
        self.root = root
        self.suffix = ['.jpg', '.png']
        self.fileList = fileList
        self.imgList = []
        self.labelList = []

    def get_file_list(self):
        for file_out in self.fileList:
            for name in os.listdir(os.path.join(self.root, file_out, 'images')):
                if os.path.splitext(name)[-1] in self.suffix:
                    labelname = name.replace('.jpg','.lines.json')
                    self.imgList.append(os.path.join(self.root, file_out, 'images',name))
                    self.labelList.append(os.path.join(self.root, file_out, 'labels',labelname))


    def move_file(self):
        cout = 0
        for idx ,txtname in enumerate(self.labelList):
            info =[]
            with open(txtname, 'r') as f:
                infoList = json.load(f)['Lines']

            if (sum([len(xx) for xx in infoList]) / len(infoList)) < 10\
                    or min([len(xx) for xx in infoList]) < 7\
                    or len(infoList) > 6 or len(infoList)<3:
                continue

            filename = cout // 500
            if cout % 500 == 0:
                print(cout)
                os.makedirs(os.path.join(self.root, 'moive', str(filename), 'image') ,exist_ok=True)
                os.makedirs(os.path.join(self.root, 'moive',str(filename), 'label') ,exist_ok=True)

            new_imgname = "1%05d.jpg"%cout
            new_txtname = "1%05d.lines.json"%cout

            shutil.copy(txtname, os.path.join(self.root, 'moive' ,str(filename), 'label',new_txtname))
            shutil.copy(self.imgList[idx], os.path.join(self.root,'moive', str(filename), 'image',new_imgname))
            cout += 1

        print('cout',cout)




    def __len__(self):
        assert len(self.imgList) == len(self.labelList)
        return len(self.imgList)


def main():
    root = '/home/wqg/data/CurveLanes'
    fileList = ['train', 'valid']
    fileList = ['valid']
    p = porcess(root,fileList)
    p.get_file_list()
    p.move_file()


    print(p.labelList[20] , p.imgList[20])




if __name__ == "__main__":
    main()


    """
    avg : 10  min :7         train:30503     val: 3476
    
    
    """