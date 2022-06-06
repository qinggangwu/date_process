from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
# import json
# import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom

from tqdm import tqdm


def poly2patch(poly2d, closed=False, alpha=1., color=None,lw =8):
    moves = {'L': Path.LINETO,
             'C': Path.CURVE4}
    points = [p[:2] for p in poly2d]
    codes = [moves[p[2]] for p in poly2d]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.CLOSEPOLY)
    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else 'none',
        edgecolor=color,  # if not closed else 'none',
        lw=lw, alpha=alpha,         # lw 大小调整线的粗细
        antialiased=False, snap=True)

def draw_lane(objects, ax):
    plt.draw()
    color = (1, 1, 1)
    for obj in objects:
        alpha = 1.0
        lw =8
        poly2d = obj[0]
        # if "双" in obj[1]:
        #     lw = 14
        ax.add_patch(poly2patch(
                poly2d, closed=False,
                alpha=alpha, color=color ,lw =lw))
    ax.axis('off')

def get_point(info ):
    value = info.getAttribute("points").split(';')
    valueintlist = []
    for _ in value:
        num = _.split(',')
        num.append("L")
        valueintlist.append(num)

    try:
        label_read = info.getElementsByTagName("attribute")[0].childNodes[0].data
    except:
        label_read = info.getAttribute("label")
    # print(valueintlist,label_read)
    return (valueintlist,label_read)


def main():
    arg = parse_args()

    for part_ in arg.partList:

        xmlpath = os.path.join(arg.dirpath, "{}/annotations.xml".format(part_))
        out_dir = os.path.join(arg.dirpath, '{}/line_mask'.format(part_))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 使用minidom解析器打开 XML 文档
        DOMTree = xml.dom.minidom.parse(xmlpath)
        collection = DOMTree.documentElement

        # 解析每张图片信息
        imgmessages = collection.getElementsByTagName("image")


        for messages_i in tqdm(range(len(imgmessages))):
            messages = imgmessages[messages_i]

            flag = True
            try:
                img_path = ''
                img_path = messages.getAttribute("name")
                img_name = img_path.replace('jpg', 'png')
                width = int(messages.getAttribute("width"))
                height = int(messages.getAttribute("height"))
            except:
                flag = False
                print('解析错误图片为: %s'%img_path)

            info = messages.getElementsByTagName('polyline')
            valueintlist = []
            for ind, labelinfo in enumerate(info):
                data = get_point(labelinfo)
                valueintlist.append(data)


            w = 16
            h = 9
            dpi = height // h
            if flag:

                fig = plt.figure(figsize=(w, h), dpi=dpi)
                ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
                out_path = os.path.join(out_dir, img_name)
                # print(out_path)
                ax.set_xlim(0, width - 1)
                ax.set_ylim(0, height - 1)
                ax.invert_yaxis()
                ax.add_patch(poly2patch(
                    [[0, 0, 'L'], [0, height - 1, 'L'],
                    [width - 1, height - 1, 'L'],
                    [width - 1, 0, 'L']],
                    closed=True, alpha=1., color=(0, 0, 0)))



                draw_lane(valueintlist, ax)   # 绘制车道线
                fig.savefig(out_path, dpi=dpi)
                plt.close()

            else:
                pass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dirpath',
                        default= '/home/wqg/data/公司数据/car_line/',
                        type= str,
                        help='图片文件所在文件夹的根目录')
    parser.add_argument('--partList',
                        default=['657'],
                        # default=['657', '658', '656', '655', '654', '653', '652'],
                        type= list,
                        help='图片文件夹名称的列表'
                        )

    # parser.add_argument('--save_clip_det_region',
    #                     default=pathdir + 'obj_det/' + part_ + '/images',
    #                     help='保存做字符区域分割的标签的文件路径'
    #                     )
    return parser.parse_args()


if __name__ == '__main__':
    main()
