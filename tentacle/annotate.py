import numpy as np
import cv2
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
from collections import defaultdict
import json
import os


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


def get_bboxes(big, small):
    '''get list of bounding box of small block in big picture
        
    Args:
        big: the big picture
        small: the small block

    Returns:
        list of bounding box(x, y, w, h)
    '''
    img_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
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
               'bndbox': list(map(lambda x: dict(zip(ss, x)), bbs))})
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
        return json.load(f)


def get_category(file):
    '''get image category from file name prefix
    
    Args:
        file: suppose the catgory is the first part 
            of file split with underscore
    
    Returns:
        the category string
    '''
    return str.split(os.path.splitext(os.path.basename(file))[0], '_')[0]


def get_cats(cats_file, small_dir):
    '''get catagories from file or dir
    Args:

    '''
    pass

def gen_annotations(big_dir, small_dir, out_file):
    '''annotate all small object in each big picture in big_dir
    
    Args:
        big_dir: dir for big pictures
        small_dir: dir for small object(category)
        out_file: output result to the json file
        
    Returns:
        None
    '''

    cache_small = {}
    for root, _, files in os.walk(small_dir):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            small = cv2.imread(full_file_name, 0)
            if small is None:
                break
            catg = get_category(file_name)
            cache_small[catg] = small

    if not cache_small:
        return

    d = edict()
    cats = [(i + 1, k) for i, k in enumerate(cache_small.keys())]
    d.categories = list(map(lambda p: {'id':p[0], 'name':p[1]}, cats))
    d.annotations = []

    for root, _, files in os.walk(big_dir):
        parent = os.path.abspath(os.path.join(big_dir, '..'))
        abs_root = os.path.abspath(root)
        rel_folder = str.strip(abs_root.replace(parent, ''), os.path.sep)

        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            big = cv2.imread(full_file_name)
            if big is None:
                break

            item = {'folder': rel_folder,
                    'filename': file_name,
                    'size': {'width': big.shape[0], 'height': big.shape[1]}}

            bbs = []
            for catg, small in cache_small.items():
                a_catg_bbs = get_bboxes(big, small)
                if len(a_catg_bbs) == 0:
                    continue
                bbs.append(bbs_to_dict(a_catg_bbs, catg))

            if bbs:
                item['object'] = bbs

            d.annotations.append(item)

    if d:
        save_to_json(d, out_file)


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


if __name__ == '__main__':
    ds = '../dataset/'
    gen_annotations(ds + 'bigpics', ds + 'atomitems', ds + 'labels.json')
#     x = np.array([513, 570, 513, 572, 512, 569, 570, 513])
#     x_ = approx_reduce(x, 2)
#     print(x)
#     print(x_)
