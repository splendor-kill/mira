# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from matplotlib import animation

import pyautogui
import win32gui
import win32con

import _init_paths
from utils.timer import Timer
from model.config import cfg
from model.test_vgg16 import im_detect
from model.nms_wrapper import nms
from nets.vgg16 import vgg16
from nets.res101 import Resnet101

from annotate import load_json
from annotate import load_int_list


def get_win_dims(name):
    hwnd = win32gui.FindWindow(None, name)    
    rect = win32gui.GetWindowRect(hwnd)
    print(rect)
    return rect, hwnd


def capture_window(x, y, w, h, hwnd):
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1)
    im = pyautogui.screenshot(region=(x, y, w, h))
    return im


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def updatefig(n, *args):
    hwnd, rect, sess, net = args[0], args[1], args[2], args[3]
    img = capture_window(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1], hwnd)
    dets, idx_cls = recognize(sess, net, np.asarray(img))
    patches = vis_detections(img, dets, idx_cls, thresh=.8)
    return tuple(patches)


def vis_detections(im, dets, idx_cls, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    COLOR_N = 30
    cmap = get_cmap(COLOR_N)
    global fig, ax
    patches = []
    patches.append(ax.imshow(im, aspect='equal', animated=True))
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        class_name = CLASSES[idx_cls[i]]
        patches.append(ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=cmap(i % COLOR_N), linewidth=3., alpha=0.5)))
        patches.append(ax.text(bbox[0] + 3, (bbox[3] + bbox[1]) / 2,
                '{:s}'.format(class_name),
                bbox=dict(facecolor='blue', alpha=0.4),
                fontsize=14, color='white'))
    return patches


cfg.DATA_DIR = './dataset'
cfg.ANCHOR_SCALES = [2.2, 4.2, 7.0, 7.6]
cfg.ANCHOR_RATIOS = [0.75, 0.83, 1.18, 1.33]

cats = load_json(os.path.join(cfg.DATA_DIR, 'cats.json'))
cat_ids = list(cats.keys())
CLASSES = [cats[i]['name'] for i in sorted(cat_ids)]
CLASSES = tuple(['__background__'] + CLASSES)
print('number of classes:', len(CLASSES))

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_20000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'mahjong': ('mj_train',)}


def recognize(sess, net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    print('img size:', im.shape)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    N = scores.shape[0]  # num of proposals
    C = scores.shape[1] - 1  # num of real classes

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    all_boxes_scores = np.zeros((N * C, 5), dtype=np.float32)

    print(len(CLASSES), all_boxes_scores.shape)

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        all_boxes_scores[N * (cls_ind - 1):N * cls_ind, :4] = cls_boxes
        all_boxes_scores[N * (cls_ind - 1):N * cls_ind, 4] = cls_scores

    keep = nms(all_boxes_scores, NMS_THRESH)
    dets = all_boxes_scores[keep, :]
    idx_cls = [i // N + 1 for i in keep]
    print('for class %d(%s) remain %d bboxes, total %d proposals' % (cls_ind, cls, len(dets), len(all_boxes_scores)))

    return dets, idx_cls


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [mahjong]',
                        choices=DATASETS.keys(), default='mahjong')
    args = parser.parse_args()
    return args


class Hud(object):

    def __init__(self, **kwargs):
        self.img = np.zeros((1,1,3))
        
        fig, ax = plt.subplots()
        imax = ax.imshow(self.img, animated=True)

        def _init():
            return imax,

        def _update(n, *args):
            imax.set_data(self.img)
            return imax,

        self.fig = fig
        self.init_func = _init
        self.update_func = _update
        
    def update_img(img):
        self.img = img
        
    def animate(self, compute_step):
        def animate_step(i):
            compute_step()
            self.update_func()

        animation.FuncAnimation(self.fig, animate_step, init_func=self.init_func, interval=50, blit=True)
        plt.show()
    

MY_DPI = 96
WIN_WIDTH, WIN_HEIGHT = 600, 320
mpl.rcParams['toolbar'] = 'None'
fig, ax = plt.subplots(figsize=(WIN_WIDTH / MY_DPI, WIN_HEIGHT / MY_DPI))
fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
plt.axis('off')
plt.tight_layout()


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True
    cfg.USE_GPU_NMS = False
    args = parse_args()

    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('./output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly? If you want something '
                       'simple and handy, try ./tools/demo_depre.py first.').format(tfmodel + '.meta'))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=tfconfig)
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = Resnet101(batch_size=1)
    else:
        raise NotImplementedError

    net.create_architecture(sess, "TEST", len(CLASSES),
                          tag='default', anchor_scales=cfg.ANCHOR_SCALES, anchor_ratios=cfg.ANCHOR_RATIOS)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    rect, hwnd = get_win_dims('欢乐麻将全集')

    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, rect[2] - rect[0], rect[3] - rect[1], 0)
    rect = win32gui.GetWindowRect(hwnd)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(rect[2], 50, WIN_WIDTH, WIN_HEIGHT)
    
    ani = animation.FuncAnimation(fig, updatefig, interval=400, blit=True, 
        fargs=(hwnd, rect, sess, net))
    plt.show()
