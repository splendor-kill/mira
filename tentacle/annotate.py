import numpy as np
import cv2
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
from collections import defaultdict
import json
import os
import time


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def key_to_int(d):
    return dict({int(k) if k.isdigit() else k: v for k, v in d.items()})


def get_bboxes(big, small):
    '''get list of bounding box of small block in big picture

    Args:
        big: the big picture, gray(2d)
        small: the small block, gray(2d)

    Returns:
        list of bounding box(x, y, w, h)
    '''
    img_gray = big if big.ndim == 2 else cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    template = small
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    bbs = []
    for pt in zip(*loc[::-1]):
        bbs.append((pt[0], pt[1], w, h))
    if len(bbs) != 0:
        bbs = reduce_boxes(bbs, 2).tolist()
    return bbs


def bbs_to_dict(bbs, label):
    '''covert list of bounding box to dict asso with label
    
    Args:
        bbs: list of bounding box
        label: the category label of small image
    '''
    ss = 'xywh'
    d = edict({'name': label,
               'bbox': list(map(lambda x: dict(zip(ss, x)), bbs))})
    return d


def save_to_json(obj, file):
    '''save json object to file
    
    Args:
        obj: json object
        file: saved file
    '''
    with open(file, 'w') as f:
        json.dump(obj, f, cls=MyEncoder)


def load_json(file):
    '''load json
    
    Args:
        file: json file
        
    Returns:
        the json object
    '''

    with open(file, 'r') as f:
        return json.load(f, object_hook=key_to_int)


def get_category(file):
    '''get image category from file name prefix
    
    Args:
        file: suppose the catgory is the first part 
            of file split with underscore
    
    Returns:
        the category string
    '''
    return str.split(os.path.splitext(os.path.basename(file))[0], '_')[0]


def get_cats(cats_file, catg_dir):
    '''get catagories from file or dir, cache for speed
    Args:
        cats_file: .json file about categories
        catg_dir: template file dir

    Returns:
        dict: categories information
    '''
    if os.path.exists(cats_file):
        return load_json(cats_file)

    # cache it if it is not exists
    d = parse_cats(catg_dir)
    save_to_json(d, cats_file)
    return d


def parse_cats(catg_dir):
    '''get all categories from a dir,
       category template file name specification:
       <catg_name>[_nnn].png
       catg_name be indicated by user or by system

    Args:
        catg_dir: template file dir

    Returns:
        dict: information about categories
    '''
    cat_names = set()
    d = {}
    next_id = 1
    for root, _, files in os.walk(catg_dir):
        rel_folder = os.path.relpath(root, os.path.dirname(catg_dir))
        for file_name in files:
            catg = get_category(file_name)
            # keep category unique
            if catg in cat_names:
                continue
            cat_names.add(catg)
            full_file_name = os.path.join(root, file_name)
            small = cv2.imread(full_file_name, 0)
            if small is None:
                break

            item = edict()
            item.name = catg
            item.folder = rel_folder
            item.file_name = file_name
            d[next_id] = item
            next_id += 1
    return d


def gen_annotations(big_dir, small_dir, cats_file, images_file, annos_file):
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

    anno_dict = {}
#     cats = [(i + 1, k) for i, k in enumerate(cache_small.keys())]
#     d.categories = list(map(lambda p: {'id':p[0], 'name':p[1]}, cats))
#     d.annotations = []

    images_dict = {}
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
                    'file_name': file_name,
                    'width': big.shape[1],
                    'height': big.shape[0]}

            images_dict[next_image_id] = item

            bbs = []
            for catg, small in cache_small.items():
                a_catg_bbs = get_bboxes(big, small)
                if len(a_catg_bbs) == 0:
                    continue
                m = {}
                m[catg] = bbs_to_dict(a_catg_bbs, catg).bbox
                anno_dict[next_image_id] = m
#                 bbs.append(bbs_to_dict(a_catg_bbs, catg))
            next_image_id += 1
#             if bbs:
#                 anno_dict
#                 item['object'] = bbs

#             d.annotations.append(item)

    if images_dict:
        save_to_json(images_dict, images_file)
    if anno_dict:
        save_to_json(anno_dict, annos_file)


def approx_reduce(x, fluctation):
    x_inc = np.sort(x)
    si = np.argsort(x, kind='mergesort')
    delta = np.diff(x_inc)
    x_keep = np.append([1], delta >= fluctation)
    remains = x_inc[x_keep > 0]
    stairs = np.cumsum(x_keep) - 1
    reduced = np.array([remains[i] for i in stairs])
    return reduced[np.argsort(si)]


def reduce_boxes(bbs, fluctation):
    '''reduce this box if it is redundant,
    that is two boxes have same position and size approximately

    Args:
        bbs: bounding boxes may have redundant boxes

    Returns:
        reduced bounding boxes
    '''
    assert len(bbs) > 0
    a = np.array(bbs)
    x = a[:, 0]
    y = a[:, 1]
    x_ = approx_reduce(x, fluctation)
    y_ = approx_reduce(y, fluctation)
    a[:, 0] = x_
    a[:, 1] = y_
    ua = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
    return ua


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def split_dataset(images_file, train_file, val_file):
    images = load_json(images_file)
    ids = list(images.keys())
    ids = np.array(ids)
    point = int(len(ids) * 0.8)
    np.random.shuffle(ids)
    train_set = ids[:point].tolist()
    val_set = ids[point:].tolist()

    save_int_list(train_set, train_file)
    save_int_list(val_set, val_file)

def save_int_list(ids, file):
    with open(file, 'w') as f:
        f.write('\n'.join(map(str, ids)))

def load_int_list(file):
    with open(file, 'r') as f:
        return list(map(int, map(str.strip, f.readlines())))

if __name__ == '__main__':
    ds = '../dataset/'
    begin = time.time()

#     get_cats(ds + 'cats.json', ds + 'atomitems')
    gen_annotations(ds + 'bigpics', ds + 'atomitems', ds + 'cats.json', ds + 'images.json', ds + 'annotations.json')
#     split_dataset(ds + 'images.json', ds + 'train.txt', ds + 'val.txt')
#     x = np.array([513, 570, 513, 572, 512, 569, 570, 513])
#     x_ = approx_reduce(x, 2)
#     print(x)
#     print(x_)
    print('time cost(s):', time.time() - begin)
