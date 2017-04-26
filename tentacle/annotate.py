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

def reduce_boxes(bbs):
    '''reduce this box if it is redundant, 
    that is two boxes have same position and size approximately
    
    Args:
        bbs: bounding boxes may have redundant boxes
        
    Returns:
        reduced bounding boxes
    '''
    return bbs 

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
    threshold = 0.9
    loc = np.where(res >= threshold)

    bbs = []
    for pt in zip(*loc[::-1]):
        bbs.append((pt[0], pt[1], w, h))

    bbs = reduce_boxes(bbs)
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

    for root, _, files in os.walk(big_dir):
        parent = os.path.abspath(os.path.join(big_dir, '..'))
        rel_folder = str.strip(root.replace(parent, ''), os.path.sep)

        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            big = cv2.imread(full_file_name)
            if big is None:
                break

            bbs = []
            for catg, small in cache_small.items():
                a_catg_bbs = get_bboxes(big, small)
                if not a_catg_bbs:
                    continue                
                bbs.append(bbs_to_dict(a_catg_bbs, catg))

            if not bbs:
                continue
            d.folder = rel_folder
            d.filename = file_name
            d.size = {'width': big.shape[0], 'height': big.shape[1]}
            d.object = bbs

    if edict:
        save_to_json(d, out_file)




if __name__ == '__main__':
    gen_annotations('/home/splendor/Pictures/mj/bigpics', '/home/splendor/Pictures/mj/atomitems', 'labels.json')
