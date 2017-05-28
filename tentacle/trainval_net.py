from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import argparse
import pprint
import numpy as np
import tensorflow as tf

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import datasets.imdb
from nets.vgg16 import vgg16
from nets.res101 import Resnet101

# from ds_mj import miradb
from ds_mj_pascal import ds_mj_pascal

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pretrained model weights',
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default=None, type=str)
    parser.add_argument('--imdbval', dest='imdbval_name',
                        help='dataset to validate on',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=10000, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default=None, type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16 or res101',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    return args


__sets = {}
for project in ['mj']:
    for split in ['train', 'val']:
        name = '{}_{}'.format(project, split)
        __sets[name] = (lambda split=split, project=project: ds_mj_pascal(project, split))
          
def get_imdb(name):
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def combined_roidb(imdb_names):
    """
    Combine multiple roidbs
    """

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return imdb, roidb

#     roidbs = [get_roidb(s) for s in imdb_names.split('+')]
#     imdb, roidb = roidbs[0]
    imdb = get_imdb(imdb_names)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
#     if len(roidbs) > 1:
#         for r in roidbs[1:]:
#             roidb.extend(r)
#         tmp = get_imdb(imdb_names.split('+')[1])
#         imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
#     else:
#         imdb = get_imdb(imdb_names)
    return imdb, roidb


def get_output_tb_dir(imdb, weights_filename):
  """overwrite model.config.get_output_tb_dir
  change dir 'tensorboard' to 'summary'
  """
  outdir = os.path.abspath(os.path.join(cfg.ROOT_DIR, 'summary', cfg.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = os.path.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


if __name__ == '__main__':
    print(os.getcwd())
    args = parse_args()
    
    print('Called with args:')
    print(args)
    
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    
    print('Using config:')
    pprint.pprint(cfg)
    
    np.random.seed(cfg.RNG_SEED)
    
    # train set
    imdb, roidb = combined_roidb(args.imdb_name)
    print('{:d} roidb entries'.format(len(roidb)))
    
    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))
    
    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))
    
    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    _, valroidb = combined_roidb(args.imdbval_name)
    print('{:d} validation roidb entries'.format(len(valroidb)))
    cfg.TRAIN.USE_FLIPPED = orgflip
    
    if args.net == 'vgg16':
        net = vgg16(batch_size=cfg.TRAIN.IMS_PER_BATCH)
    elif args.net == 'res101':
        net = Resnet101(batch_size=cfg.TRAIN.IMS_PER_BATCH)
    else:
        raise NotImplementedError
    
    train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=args.weight,
              max_iters=args.max_iters)
