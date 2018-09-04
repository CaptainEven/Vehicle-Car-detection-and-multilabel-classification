# coding: utf-8

import os
import shutil


def merge_violet2blue(train_root):
    """
    to simplify data labeling and training, we merge violet into blue
    """
    sub_dirs = [train_root + '/' + x for x in os.listdir(train_root)]
    violets = [x for x in sub_dirs if os.path.isdir(x) and 'Violet' in x]
    print('=> violets:\n', violets)

    # merge violet files into blue
    for x in violets:
        dst_dir = x.replace('Violet', 'Blue')

        if os.path.exists(dst_dir):
            for y in os.listdir(x):
                src_path = x + '/' + y
                if os.path.exists(src_path):
                    dst_path = dst_dir + '/' + y
                    if not os.path.exists(dst_path):
                        shutil.copy(src_path, dst_dir)

    # delete empty sub-dirs which contains violet
    for x in violets:
        if os.path.exists(x):
            shutil.rmtree(x)


def viz_err(err_path, root='f:/'):
    """
    visualize the detailed err info.
    """
    err_dict = pickle.load(open(err_path, 'rb'))
    # print(err_dict)

    fig = plt.figure()  #

    for k, v in err_dict.items():
        img_path = root + k
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            plt.gcf().set_size_inches(8, 8)
            plt.imshow(img)
            plt.title(img_path + '\n' + v)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.show()


if __name__ == '__main__':
    # merge_violet2blue('f:/vehicle_test')
    pass
