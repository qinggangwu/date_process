#----------------------------------------------#
#------修改图片和标签的名字，为了符合psenet工程编码---#
#----------------------------------------------#

import argparse
import os
import shutil

class ProcessTxt:
    def __init__(self, args):
        self.img_dir = args.img_dir #txt dir
        self.txt_dir = args.txt_dir

    def loadAllTagFile(self, dir,tag):
        result = []
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.splitext(file_path)[1]==tag:
                result.append(file_path)
        return result


    def process(self):
        img_names = self.loadAllTagFile(self.img_dir, '.jpg')
        i_num = 0 #8816
        for img_ in img_names:
            i_num += 1
            print(str(len(img_names)) + ':' + str(i_num))
            imgname = os.path.basename(img_)
            img_file = os.path.splitext(imgname)[0]
            txt_file = os.path.join(self.txt_dir, img_file + '.txt')

            new_img_name = os.path.join(self.img_dir, 'img_'+str(i_num)+'.jpg')
            new_txt_name = os.path.join(self.txt_dir,'gt_'+'img_'+str(i_num)+'.txt')
            shutil.move(img_, new_img_name)
            shutil.move(txt_file, new_txt_name)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',
                        default='/home/fuxueping/sdb/data/container/container_det/630/test/img',
                        help='图片文件所在文件夹')
    parser.add_argument('--txt_dir',
                        default='/home/fuxueping/sdb/data/container/container_det/630/test/gt',
                        help='标签文件所在文件夹'
                        )

    return parser.parse_args()

def main(args):
    G = ProcessTxt(args)
    G.process()


if __name__ == '__main__':
    main(parse_args())
