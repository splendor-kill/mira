import numpy as np
import cv2
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
from collections import defaultdict
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
import os
import time
import datetime
import random
from annotate import load_json
from annotate import get_cats
from annotate import get_bboxes, bbs_to_dict


def check_a_file(xml_file, data_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    folder = root.find('folder').text
    img_file = root.find('filename').text
    img = cv2.imread(os.path.join(data_dir, folder, img_file))
    objs = root.findall('object')
    for obj in objs:
        cat = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(img, cat, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('show', img)
    cv2.waitKey(0)


def save_to_xml(obj, file):
    root = Element('annotation')
    SubElement(root, 'folder').text = obj['folder']
    SubElement(root, 'filename').text = obj['filename']
    size_node = SubElement(root, 'size')
    SubElement(size_node, 'width').text = str(obj['width'])
    SubElement(size_node, 'height').text = str(obj['height'])

    for d in obj['object']:
        for box in d.bbox:
            object_node = SubElement(root, 'object')
            SubElement(object_node, 'name').text = d.name
            bndbox_node = SubElement(object_node, 'bndbox')
            SubElement(bndbox_node, 'xmin').text = str(box.x)
            SubElement(bndbox_node, 'ymin').text = str(box.y)
            SubElement(bndbox_node, 'xmax').text = str(box.x + box.w)
            SubElement(bndbox_node, 'ymax').text = str(box.y + box.h)

#     print(ET.tostring(root))
    tree = ET.ElementTree(root)
    tree.write(file)


def gen_annotations(big_dir, small_dir, cats_file, prefix, annos_dir):
    '''annotate all small object in each big picture in big_dir
    
    Args:
        big_dir: dir for big pictures
        small_dir: dir for small object(category)
        cats_file: information about categories
        out_file: output result to the json file
        
    Returns:
        None
    '''

    cache_small = {}

    cats = get_cats(cats_file, small_dir)
    for cat_id, val in cats.items():
        full_file_name = os.path.join(os.path.dirname(small_dir), val['folder'], val['file_name'])
        small = cv2.imread(full_file_name, 0)
        if small is None:
            break
        cache_small[cat_id] = small

    if not cache_small:
        return


    if not os.path.exists(annos_dir):
        os.makedirs(annos_dir)

    next_image_id = 1
    for root, _, files in os.walk(big_dir):
#         parent = os.path.abspath(os.path.join(big_dir, '..'))
#         abs_root = os.path.abspath(root)
        rel_folder = os.path.relpath(root, os.path.dirname(big_dir))
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            big = cv2.imread(full_file_name, 0)
            if big is None:
                break

            item = {'folder': rel_folder,
                    'filename': file_name,
                    'width': big.shape[1],
                    'height': big.shape[0]}

            bbs = []
            for catg, small in cache_small.items():
                a_catg_bbs = get_bboxes(big, small)
                if len(a_catg_bbs) == 0:
                    continue
                bbs.append(bbs_to_dict(a_catg_bbs, cats[catg]['name']))
            item['object'] = bbs

            new_file_name = '%s_%05d' % (prefix, next_image_id)
            save_to_xml(item, os.path.join(annos_dir, new_file_name + '.xml'))

            os.rename(full_file_name, os.path.join(root, new_file_name + os.path.splitext(file_name)[-1]))

            next_image_id += 1


def split_dataset(annos_dir, image_sets_dir):
    ids = []
    for f in os.listdir(annos_dir):
        ids.append(os.path.splitext(f)[0])
    random.shuffle(ids)
    point = int(len(ids) * 0.8)
    train_set = ids[:point]
    val_set = ids[point:]

    save_id_list(train_set, os.path.join(image_sets_dir, 'train.txt'))
    save_id_list(val_set, os.path.join(image_sets_dir, 'val.txt'))


def save_id_list(ids, file):
    with open(file, 'w') as f:
        f.write('\n'.join(ids))


def load_id_list(file):
    with open(file, 'r') as f:
        return list(map(str.strip, f.readlines()))


def save_to_xml2(info, bbs, cats, new_file_name, file):
    root = Element('annotation')
    SubElement(root, 'folder').text = info['folder']
    SubElement(root, 'filename').text = new_file_name
    size_node = SubElement(root, 'size')
    SubElement(size_node, 'width').text = str(info['width'])
    SubElement(size_node, 'height').text = str(info['height'])

    for catg, bb in bbs.items():
        for box in bb['bbox']:
            object_node = SubElement(root, 'object')
            SubElement(object_node, 'name').text = cats[catg]['name']
            bndbox_node = SubElement(object_node, 'bndbox')
            SubElement(bndbox_node, 'xmin').text = str(box['x'])
            SubElement(bndbox_node, 'ymin').text = str(box['y'])
            SubElement(bndbox_node, 'xmax').text = str(box['x'] + box['w'])
            SubElement(bndbox_node, 'ymax').text = str(box['y'] + box['h'])

    tree = ET.ElementTree(root)
    tree.write(file)


def convert_to_pascal_from_coco(big_dir, annos_json, images_json, cats_json, prefix, annos_dir):
    cats = load_json(cats_json)
    images = load_json(images_json)
    annos = load_json(annos_json)

    if not os.path.exists(annos_dir):
        os.makedirs(annos_dir)

    i = 0
    for k, v in annos.items():
        info = images[k]
        file_name = info['file_name']

        new_file_name_stem = '%s_%05d' % (prefix, int(k))
        new_file_name = new_file_name_stem + os.path.splitext(file_name)[-1]
        save_to_xml2(info, v, cats, new_file_name, os.path.join(annos_dir, new_file_name_stem + '.xml'))

        old_file = os.path.join(big_dir, file_name)
        new_file = os.path.join(big_dir, new_file_name)
        os.rename(old_file, new_file)
        i += 1
    print('total %d files converted' % (i,))


if __name__ == '__main__':
    ds = '../dataset/'
    begin = time.time()

#     gen_annotations('/home/splendor/jumbo/annlab/mj/bigpics_test',
#                     ds + 'atomitems', ds + 'cats.json', 'qqmj', ds + 'anno_pascal')
    split_dataset(ds + 'ds_mj/annotations', ds + 'ds_mj/sets')
#     convert_to_pascal_from_coco(ds + 'bigpics',
#                                 ds + 'annotations.json',
#                                 ds + 'images.json',
#                                 ds + 'cats.json',
#                                 'qqmj',
#                                 ds + 'anno_pascal')

    print('time cost(s):', time.time() - begin)
