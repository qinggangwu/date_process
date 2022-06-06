'''
将数据的标签通过相应的图片名拷贝出来
'''
import os
import shutil
if __name__=='__main__':
    img_dir = '/home/fuxueping/sdb/data/container/container_obj_det/make_data/seg_label_single_charcter/无底黑字/无底黑字_单个字符'
    label_dir = '/home/fuxueping/sdb/data/container/container_obj_det/make_data/seg_label_single_charcter/无底黑字/fintune_txt'
    save_label_dir = '/home/fuxueping/sdb/data/container/container_obj_det/make_data/seg_label_single_charcter/无底黑字/无底黑字_单个字符_label'
    img_paths = os.listdir(img_dir)
    for img_path in img_paths:
        # img_path = '-nfs-集装箱项目-集装箱2387-20191217_122006_8.jpg'
        txt_path = os.path.join(label_dir, os.path.basename(img_path)[:-4] + '.txt')
        print(img_path)
        if os.path.exists(txt_path):
            result_line = ''
            alllines = open(txt_path, 'r', encoding='utf-8').readlines()
            for line in alllines:
                if len(line.strip()) != 0:
                    path = line.strip().split('\t')[0]
                    label = line.strip().split('\t')[1]
                    new_line = path+','+label+'\n'
                    result_line += new_line
            save_label_path = os.path.join(save_label_dir, os.path.basename(img_path)[:-4] + '.txt')
            f_save = open(save_label_path, 'w', encoding='utf-8')
            f_save.write(result_line)
            # shutil.copy(txt_path, save_label_path)
        else:
            print(os.path.basename(img_path)[:-4] + '.txt', 'is miss')
