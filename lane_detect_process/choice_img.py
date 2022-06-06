import os
import random
import shutil

def choice_img():
    root = '/home/wqg/data/tusimple/clips'
    root = '/home/wqg/data/CULane'
    os.makedirs(root+'/choice_img',exist_ok= True )

    flieList = os.listdir(root)
    flieList = ['driver_193_90frame', 'laneseg_label_w16',  'driver_37_30frame', 'driver_23_30frame', 'driver_182_30frame', 'driver_100_30frame', 'driver_161_90frame']

    cont = 0
    for file in flieList:
        dir = os.path.join(root,file)
        for imgfile in random.choices(os.listdir(dir),k=100):
            imgList = os.listdir(os.path.join(dir,imgfile))
            if len(imgList)>0:
                imgname = random.choice(imgList)
                if os.path.splitext(imgname)[-1] =='.jpg':
                    new_name = "%06d.jpg"%cont
                    cont +=1
                    # print(imgname,new_name)

                    shutil.copy(os.path.join(dir,imgfile,imgname), os.path.join(root+'/choice_img',new_name) )


def read_txt():
    with open('/home/wqg/data/tusimple/train.txt','r') as f:
        txtread = f.readlines()

    train= []
    val = []
    for idx ,line in enumerate(txtread):
        if idx %10 ==0:
            val.append(line)
        else:
            train.append(line)
    with open('/home/wqg/data/tusimple/train.txt','w') as ta:
        ta.writelines(train)
    with open('/home/wqg/data/tusimple/testval.txt','w') as te:
        te.writelines(val)


if __name__ == "__main__":
    read_txt()