from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json


lib_path = '/home/splendor/learn/tf-faster-rcnn/lib'

if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.config import cfg

import annotate

class MIRADSAPI:
    def __init__(self, project, image_set, annotation_file, cats_file, images_file):
        self._project = project
        self._image_set = image_set
        self._annos_file = annotation_file
        self._cats_file = cats_file
        self._images_file = images_file


    def loadCats(self):
        cats = {}
        d = annotate.load_json(self._cats_file)
        for k, v in d:
            cats[k] = v['name']
        return cats

    def getCatIds(self):
        if self._cats is None:
            self._cats = self.loadCats()
        return list(self._cats.keys())

    def getImgIds(self):
        if self._images is None:
            self._images = annotate.load_json(self._images_file)
        return list(self._images.keys())

    def loadAnns(self, image_index):
        if self._annos is None:
            self._annos = annotate.load_json(self._annos_file)

        anno = self._annos['%s' % image_index]
        return anno


class miradb(imdb):
  def __init__(self, project, image_set, year):
    imdb.__init__(self, project + '_' + image_set)
    # specific config options
    self.config = {}
    # name, paths
    self._year = year
    self._image_set = image_set
    self._data_path = osp.join(cfg.DATA_DIR, project)
    self._DSAPI = DSAPI(self._get_ann_file())
    cats = self._DSAPI.loadCats(self._DSAPI.getCatIds())
    self._classes = tuple(['__background__'] + [c['name'] for c in cats])
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._class_to_coco_cat_id = dict([(v, k) for k, v in cats.items()])
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self.set_proposal_method('gt')
    self.competition_mode(False)

    # Some image sets are "views" (i.e. subsets) into others.
    # For example, minival2014 is a random 5000 image subset of val2014.
    # This mapping tells us where the view's images and proposals come from.
    self._view_map = {
      'minival2014': 'val2014',  # 5k val2014 subset
      'valminusminival2014': 'val2014',  # val2014 \setminus minival2014
      'test-dev2015': 'test2015',
    }
    coco_name = image_set + year  # e.g., "val2014"
    self._data_name = (self._view_map[coco_name]
                       if coco_name in self._view_map
                       else coco_name)
    # Dataset splits that have ground-truth annotations (test splits
    # do not have gt annotations)
    self._gt_splits = ('train', 'val', 'minival')

  def _get_ann_file(self):
    prefix = 'instances' if self._image_set.find('test') == -1 \
      else 'image_info'
    return osp.join(self._data_path, 'annotations',
                    prefix + '_' + self._image_set + self._year + '.json')

  def _load_image_set_index(self):
    """
    Load image ids.
    """
    image_ids = self._DSAPI.getImgIds()
    return image_ids

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    img_info = self._images[index]
    image_path = osp.join(self._data_path, img_info.folder, img_info.file_name)
    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path


  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_coco_annotation(index)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

    def _load_coco_annotation(self, index):
        img_info = self._images[index]
        width = img_info['width']
        height = img_info['height']

        objs = self._DSAPI.loadAnns(index)

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        ix = 0
        for cat_id, bbs in objs.items():
            cls = coco_cat_id_to_class_ind[cat_id]
            for bb in bbs:
                x1 = bb['x']
                y1 = bb['y']
                w = bb['w']
                h = bb['h']
                x2 = x1 + w
                y2 = y1 + h
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = w * h
                ix += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

  def _get_widths(self):
    return [r['width'] for r in self.roidb]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}

      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def _coco_results_one_category(self, boxes, cat_id):
    results = []
    for im_ind, index in enumerate(self.image_index):
      dets = boxes[im_ind].astype(np.float)
      if dets == []:
        continue
      scores = dets[:, -1]
      xs = dets[:, 0]
      ys = dets[:, 1]
      ws = dets[:, 2] - xs + 1
      hs = dets[:, 3] - ys + 1
      results.extend(
        [{'image_id': index,
          'category_id': cat_id,
          'bbox': [xs[k], ys[k], ws[k], hs[k]],
          'score': scores[k]} for k in range(dets.shape[0])])
    return results

  def _write_coco_results_file(self, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                       self.num_classes - 1))
      coco_cat_id = self._class_to_coco_cat_id[cls]
      results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                     coco_cat_id))
    print('Writing results json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
      json.dump(results, fid)


  def evaluate_detections(self, all_boxes, output_dir):
    res_file = osp.join(output_dir, ('detections_' +
                                     self._image_set +
                                     self._year +
                                     '_results'))
    if self.config['use_salt']:
      res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    self._write_coco_results_file(all_boxes, res_file)
    # Only do evaluation on non-test sets
    if self._image_set.find('test') == -1:
      self._do_detection_eval(res_file, output_dir)
    # Optionally cleanup results json file
    if self.config['cleanup']:
      os.remove(res_file)

