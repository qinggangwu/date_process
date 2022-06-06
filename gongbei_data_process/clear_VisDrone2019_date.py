import os


def find_truck_data():
    filedir = '/home/wqg/data/VisDrone2019/'

    file_list = os.listdir(filedir)
    for file in file_list:
        if os.path.isdir( os.path.join(filedir,file)):
            for labelfile in os.listdir(os.path.join(filedir,file,'annotations')):
                if os.path.splitext(labelfile)[-1] == ".txt":
                    with open(os.path.join(filedir,file,'annotations',labelfile) ,'r') as f:
                        infolist = f.readlines(f)
                        for info in infolist:
                            info = info.split(",")
                            if info[5] == 6:
                                print(labelfile)




if __name__ =="__main__":
    find_truck_data()