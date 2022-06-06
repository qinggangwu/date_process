import os
import json
import shutil

def get_flag(json_path):
    # json_path = '/home/wqg/data/BDD100K/datasets/det_annotations/val/a9c8091c-c67ba906.json'

    opList = ['truck','bus']
    num = 0
    with open(json_path,'r',encoding='utf-8') as fo:
        info = json.load(fo)
        objects = info['frames'][0]['objects']
        for ob in objects:
            if ob['category'] in opList:
                num +=1
                if num >=3:
                    return True

    return False


def chioce_img():
    jsonDir = '/home/wqg/data/BDD100K/datasets'

    os.makedirs(os.path.join(jsonDir,'chioce'),exist_ok=True)

    filelist = ['train','val']
    n = 0
    for files in filelist:
        jsonlist = os.listdir(os.path.join(jsonDir,'det_annotations',files))
        for jsonname in jsonlist:
            jsonpath = os.path.join(jsonDir,'det_annotations',files,jsonname)
            if get_flag(jsonpath):
                imgname = jsonname[:-4] + 'jpg'
                imgpath = os.path.join(jsonDir,'images',files,imgname)
                img_new_path = os.path.join(jsonDir,'chioce',imgname)
                n+=1
                shutil.copy(imgpath,img_new_path)

    print(n)



def main():
    chioce_img()


if __name__ == "__main__":
    main()