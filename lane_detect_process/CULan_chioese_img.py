import os
import cv2
from tqdm import tqdm
import shutil


class porcess:
    def __init__(self,root):
        self.root = root
        self.suffix = ['.jpg', '.png']
        self.imgList = []
        self.labelList = []




    def get_file_list(self):
        fileList = [ 'driver_193_90frame',  'driver_37_30frame', 'driver_23_30frame',
                     'driver_182_30frame', 'driver_100_30frame',  'driver_161_90frame']

        for file_out in fileList:
            for file_in in os.listdir(os.path.join(self.root, file_out)):
                for name in os.listdir(os.path.join(self.root, file_out, file_in)):
                    if os.path.splitext(name)[-1] in self.suffix:
                        txtname = name.replace('.jpg','.lines.txt')

                        self.imgList.append(os.path.join(self.root, file_out, file_in,name))
                        self.labelList.append(os.path.join(self.root, file_out, file_in,txtname))

    def move_file(self):
        cout = 0
        for idx ,txtname in enumerate(self.labelList):
            info =[]
            with open(txtname , 'r') as f:
                infoList = f.readlines()
            if len(infoList) <3 :
                continue
            if min([len(xx.split()) for xx in infoList]) < 40:
                continue
            # print(min([len(xx.split()) for xx in infoList]))

            filename = cout // 500
            if cout % 500 == 0:
                print(cout)
                os.makedirs(os.path.join(self.root, 'moive', str(filename), 'image') ,exist_ok=True)
                os.makedirs(os.path.join(self.root, 'moive',str(filename), 'label') ,exist_ok=True)

            new_imgname = "%06d.jpg"%cout
            new_txtname = "%06d.lines.txt"%cout

            # shutil.copy(txtname, os.path.join(self.root, 'moive' ,str(filename), 'label',new_txtname))
            # shutil.copy(self.imgList[idx], os.path.join(self.root,'moive', str(filename), 'image',new_imgname))
            cout += 1

        print('cout',cout)




    def __len__(self):
        assert len(self.imgList) == len(self.labelList)
        return len(self.imgList)


def main():
    root = '/home/wqg/data/CULane'
    p = porcess(root)
    p.get_file_list()
    p.move_file()


    print(p.labelList[20] , p.imgList[20])




if __name__ == "__main__":
    main()

    """
    if min([len(xx.split()) for xx in infoList]) < 30:  93411
    if min([len(xx.split()) for xx in infoList]) < 35:  65026
    if min([len(xx.split()) for xx in infoList]) < 38:  54036
    if min([len(xx.split()) for xx in infoList]) < 40:  43514
    
    
    """