import os
import numpy as np
import pandas as pd


def read_image(src_path):
    import cv2

    imgs = {}
    for f in os.listdir(src_path):
        img = cv2.imread(os.path.join(src_path, f))
        print(f)
        img = cv2.resize(img, (224, 224))
        name = f.replace('.jpg', '')
        imgs[name] = img
    return imgs


def read_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.fillna(0, inplace=True)
    res = df.values
    return res


def parse_label_file(src_path, sep=' ', skip_header=False):
    res = {}

    with open(src_path, 'r') as f:
        f_lines = f.readlines()
        if skip_header:
            f_lines = f_lines[1:]
        for line in f_lines:
            lines = line.split(sep)
            if len(lines) > 1:
                res[lines[0]] = lines[1].strip('\n')
    return res


def match_data_label(label_dict, imgs_dict, aux_dict=None):
    imgs, aux_data = [], []
    p = []
    for f in os.listdir('/home/xujingning/ocean/ocean_data/problem/'):
        p.append(f.split('.')[0])
        print(f.split('.')[0])
    for pic in p:
        del label_dict[pic]
    for key in label_dict.keys():
            if key in imgs_dict:
                imgs.append(imgs_dict[key])
            else:
                print('%s not in image dict' % key)
            if aux_dict is not None and key in aux_dict:
                aux_data.append(aux_dict[key])
            elif aux_dict is not None:
                print('%s not in aux data dict' % key)
    return imgs, aux_data


def convert_data_type(data, dtype=np.float32):
    if data is None:
        return data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    return data.astype(dtype)


def get_aux_data(data_path):
    if data_path is None:
        return None
    raw_aux_data = {}
    csv_data = read_csv(data_path)
    for row in csv_data[1:]:
        raw_aux_data[str(int(row[0]))] = [float(r) for r in row[1:]]
    return raw_aux_data


def get_data(data_path, label_path, aux_data_path=None):
    label_dict = parse_label_file(label_path, ',', True)
    imgs_dict = read_image(data_path)
    aux_data_dict = get_aux_data(aux_data_path)

    imgs, aux_data = match_data_label(label_dict, imgs_dict, aux_data_dict)
    imgs = convert_data_type(imgs, np.float32)
    aux_data = convert_data_type(aux_data, np.float32)
    labels = convert_data_type(label_dict.values(), np.int32)

    return labels, imgs, aux_data
