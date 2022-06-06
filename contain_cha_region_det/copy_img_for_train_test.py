'''
将划分的训练样本和测试样本拷贝，进行模型训练
'''
import argparse
import os
from tqdm import tqdm
import shutil

def mkdir_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def main(opt):

    for name in opt.names_num_list:
        # print(name)
        img_dir_train = os.path.join(opt.img_dir, name + '/train/images')
        txt_dir_train = os.path.join(opt.img_dir, name + '/train/labels')

        img_dir_test = os.path.join(opt.img_dir, name + '/test/images')
        txt_dir_test = os.path.join(opt.img_dir, name + '/test/labels')


        save_train_img = os.path.join(opt.save_dir_train, 'images')
        mkdir_dir(save_train_img)
        save_train_labels = os.path.join(opt.save_dir_train, 'labels')
        mkdir_dir(save_train_labels)

        save_test_img = os.path.join(opt.save_dir_test, 'images')
        mkdir_dir(save_test_img)
        save_test_labels = os.path.join(opt.save_dir_test, 'labels')
        mkdir_dir(save_test_labels)


        imgs_train = os.listdir(img_dir_train)
        imgs_test = os.listdir(img_dir_test)

        print(name, 'train')
        for img_i in tqdm(range(len(imgs_train))):
            img = imgs_train[img_i]
            img_path = os.path.join(img_dir_train, img)
            txt_path = os.path.join(txt_dir_train, img[:-4] + '.txt')

            save_img_path = os.path.join(save_train_img, img)
            if not os.path.exists(save_img_path):
                shutil.copy(img_path, save_img_path)
                save_txt_path = os.path.join(save_train_labels, img[:-4] + '.txt')
                shutil.copy(txt_path, save_txt_path)
            else:
                print(save_img_path)

        print(name, 'test')
        for img_i in tqdm(range(len(imgs_test))):
            img = imgs_test[img_i]

            img_path = os.path.join(img_dir_test, img)
            txt_path = os.path.join(txt_dir_test, img[:-4] + '.txt')

            save_img_path = os.path.join(save_test_img, img)
            if not os.path.exists(save_img_path):
                shutil.copy(img_path, save_img_path)
                save_txt_path = os.path.join(save_test_labels, img[:-4] + '.txt')
                shutil.copy(txt_path, save_txt_path)
            else:
                print(save_img_path)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # names_num_list = ['1777', '958', '959', '960', '961', '962', '968', '969', '970', '971', '972', '1791', '1792', '1793']
    names_num_list = ['958']
    parser.add_argument('--img_dir', type=str,
                        default='/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/container_rec', help='')
    parser.add_argument('--names_num_list', default=names_num_list, help='')
    parser.add_argument('--save_dir_train', type=str,
                        default='/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/container_rec/train', help='')

    parser.add_argument('--save_dir_test', type=str,
                        default='/media/fuxueping/7292a4b1-2584-4296-8caf-eb9788c2ffb9/data/container/container_rec/test',
                        help='')


    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
