import os
import numpy as np
import cv2
from easydict import EasyDict as edict

from annotate import get_bboxes
from annotate import load_json

cfg = edict()

def browse(file_dir, template=None):
    '''browse pics under a dir and draw box over the template
    
    Args:
        file_dir: pic dir
        template: 2d ndarray
    
    Returns:
        True if press 'q'
    '''
    for root, _, files in os.walk(file_dir):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            image = cv2.imread(full_file_name)
            if image is None:
                continue
            
            if template is not None:
                bbs = get_bboxes(image, template)
                for x, y, w, h in bbs:
                    print('bbox:', x, y, w, h)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                print()

            cv2.imshow('pic browser', image)
            k = cv2.waitKey(0)
            if k == ord('q'):                
                return True
    return False


def browse_all(file_dir, cats):
    smalls = {}
    for k, v in cats.items():
        full_name = os.path.join(cfg.data_dir, v['folder'], v['file_name'])        
        img = cv2.imread(full_name, 0)
        if img is not None:
            smalls[k] = img

    for k, v in smalls.items():
        print('broswering', cats[k]['name'])
        done = browse(file_dir, v)
        if done:
            break

    
if __name__ == '__main__':
    cfg.data_dir = '../dataset'
#     template = cv2.imread(os.path.join(cfg.data_dir, 'atomitems/b8.png'), 0)
#     browse(os.path.join(cfg.data_dir, 'bigpics'), template)
    cats = load_json(os.path.join(cfg.data_dir, 'cats.json'))
    browse_all(os.path.join(cfg.data_dir, 'bigpics'), cats)
