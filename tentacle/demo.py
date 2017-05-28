from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cv2
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

import _init_paths
from utils.timer import Timer
from model.config import cfg
from model.test_vgg16 import im_detect
from model.nms_wrapper import nms
from nets.vgg16 import vgg16
from nets.res101 import Resnet101

from annotate import load_json
from annotate import load_int_list


cfg.DATA_DIR = '../dataset'
# cfg.TEST.RPN_POST_NMS_TOP_N = 400
# cfg.ANCHOR_SCALES = [3,7,11]
# cfg.ANCHOR_RATIOS = [0.75, 1, 1.33]
# cfg.ANCHOR_SCALES = [2,7,12]
# cfg.ANCHOR_RATIOS = [0.5, 1, 2]
cfg.ANCHOR_SCALES = [6, 12, 18]
cfg.ANCHOR_RATIOS = [0.67, 0.83, 1.2, 1.5]


cats = load_json(os.path.join(cfg.DATA_DIR, 'cats.json'))
cat_ids = list(cats.keys())
CLASSES = [cats[i]['name'] for i in sorted(cat_ids)]
CLASSES = tuple(['__background__'] + CLASSES)
print('number of classes:', len(CLASSES))
print(cats)

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_10000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'mahjong': ('mj_train',)}



def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    COLOR_N = 30
    cmap = get_cmap(COLOR_N)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=cmap(i % COLOR_N), linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def vis_detections_all(im, dets, idx_cls, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    COLOR_N = 30
    cmap = get_cmap(COLOR_N)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        class_name = CLASSES[idx_cls[i]]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=cmap(i % COLOR_N), linewidth=3., alpha=0.4)
            )
        ax.text(bbox[0] + 3, (bbox[3] + bbox[1]) / 2,
                '{:s}'.format(class_name),
                bbox=dict(facecolor='blue', alpha=0.4),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    N = scores.shape[0]  # num of proposals
    C = scores.shape[1] - 1  # num of real classes

    # Visualize detections for each class
    CONF_THRESH = 0.5
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

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print('for class %d(%s) remain %d bboxes' % (cls_ind, cls, len(keep)))
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

#     keep = nms(all_boxes_scores, NMS_THRESH)
#     dets = all_boxes_scores[keep, :]
#     idx_cls = [i // N + 1 for i in keep]
#     print('for class %d(%s) remain %d bboxes, total %d proposals'%(cls_ind, cls, len(dets), len(all_boxes_scores)))
#     vis_detections_all(im, dets, idx_cls, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [mahjong]',
                        choices=DATASETS.keys(), default='mahjong')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('../output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly? If you want something '
                       'simple and handy, try ./tools/demo_depre.py first.').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = Resnet101(batch_size=1)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", len(CLASSES),
                          tag='default', anchor_scales=[6, 12, 18], anchor_ratios=[0.67, 0.83, 1.2, 1.5])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

#     im_names = ['mj/tianfeng/2.png']
#     im_names = ['mj/test_var/1_8.png']
#     im_names = ['mj/mjatlas_uni.png']
#     im_names = ['mj/mj_qqhl/qqmj_00001.png']
    im_names = ['tube/aa.png']
#     im_names = ['tube/mjscreen_373.png', 'tube/mjscreen_663.png', 'tube/mjscreen_385.png']
#     im_names = ['tube/1.jpg', 'tube/2.jpg', 'tube/3.jpg']
#     im_names = ['mj/mj_own_qqhl/mjscreen_100971.png']

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for dataset/{}'.format(im_name))
        demo(sess, net, im_name)

    plt.show()
