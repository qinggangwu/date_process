"""
集装箱的字符识别数据的标签处理：将数据标签从cvat中读取出来，按照一张图片一个txt文件进行保存
"""

from xml.dom.minidom import parse
import xml.dom.minidom
import os
import shutil
from tqdm import tqdm

class datastru:
    def __init__(self):
        self.img_path = ''
        self.height   = 0.0
        self.width    = 0.0
        self.label    = ''
        self.points   = []
        self.flag_over = bool


if __name__=='__main__':
#     part_ = '935_936'
#     part_all = ['950','951','952','953','954','955','956','963','964','965','966','967','973','974','975','998']
#     part_all =['1777', '1722', '972', '971', '970', '969', '968', '962', '961', '960', '959', '958', '940', '939', '938', '937', '936', '935', '934', '933', '928', '925', '924', '923', '922', '892', '891', '630']
    part_all = ['1793', '1792', '1791']
    for part_ in part_all:
        pathdir = '/home/jovyan/data-vol-1/wqg/container/original_data/'
        xmlpath = pathdir +part_ + '/annotations.xml'
        save_txtpath = pathdir +part_+'/labels'
        not_care_lable = '###'
        if not os.path.exists(save_txtpath):
            os.makedirs(save_txtpath)
        # 使用minidom解析器打开 XML 文档
        DOMTree = xml.dom.minidom.parse(xmlpath)
        collection = DOMTree.documentElement

        # 在集合中获取所有图片的信息
        imgmessages = collection.getElementsByTagName("image")

        # 解析每一张图片
        # allImgmessage = []
        oneImg = datastru()
        for messages_i in tqdm(range(len(imgmessages))):
            messages = imgmessages[messages_i]
            # try:
#             print("*****img*****")
            # oneImg.flag_over = False
#             if messages.hasAttribute("id"):
#                 print("id: %s" % messages.getAttribute("id"))
                # oneImg.id = messages.getAttribute("id")
            # except:
            #     import pdb;pdb.set_trace()

            if messages.hasAttribute("name"):
                # print("name: %s" % messages.getAttribute("name"))
                oneImg.img_path = messages.getAttribute("name")
#                 print(oneImg.img_path)
            pathTXT = os.path.basename(oneImg.img_path)
            pathTXT_ = os.path.splitext(pathTXT)[0]


            if messages.hasAttribute("width"):
                # print("width: %s" % messages.getAttribute("width"))
                oneImg.width = int(messages.getAttribute("width"))

            if messages.hasAttribute("height"):
                # print("name: %s" % messages.getAttribute("height"))
                oneImg.height = int(messages.getAttribute("height"))


            type = messages.getElementsByTagName('polygon')
            str_label = ''
            for i in range(len(type)):
                label_point = {}
                # if type[i].hasAttribute("label"):
                    # print("label: %s" % type[i].getAttribute("label"))
                # if type[i].hasAttribute("points"):
                    # print("points: %s" % type[i].getAttribute("points"))

                value = type[i].getAttribute("points").split(';')
                valueintlist = []

                for _ in value:
                    num = _.split(',')
                    intnums = []
                    for intnum in num:
                        intnum = round(float(intnum))
                        str_label += (str(intnum)+',')
                        intnums.append(intnum)

                    valueintlist.append(intnums)
                str_label = str_label[:-1] + '\t'

                label_read = type[i].getAttribute("label")
                if label_read == '集装箱框':#'textbox':
                    try:
                        labelvalue = type[i].getElementsByTagName("attribute")[0]
                        new_data = labelvalue.childNodes[0].data
                        label = new_data
                    except:
                        label = 'unknown' #用于处理有些标签标注漏掉的数据

                elif  '外框' in label_read:
                    # labelvalue = type[i].getElementsByTagName("attribute")[0]
                    # new_data = labelvalue.childNodes[0].data
                    label = label_read

                    # except:
                elif '忽略' in label_read:
                    # print(oneImg.img_path)
                    label = not_care_lable

                else:
                    print('label error!', label_read)
                label_point[label] = valueintlist
                str_label += label+'\n'

                    # print('end')
                # else:
                #     oneImg.flag_over = True
                #     label = type[i].getAttribute("label")
                #     label_point[label] = value
                #     str_label += label + '\n'

                # oneImg.points.append(label_point)
            # if oneImg.flag_over == False:
            #     print('这个图片没有处理完')
            # else:
            txt_file = os.path.join(save_txtpath, pathTXT_ + '.txt')
            if os.path.exists(txt_file):
                os.remove(txt_file)
            f1 = open(txt_file, 'x')
            f1.write(str_label)
            f1.close()
        print('end')

