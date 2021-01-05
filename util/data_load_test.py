from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
# from scipy.misc import imageio
from scipy import misc
# 由于scipy版本问题，使用imageio读取图像
import imageio

from IPython import embed


# from util.utils import mkdir_if_missing, write_json, read_json

class data_reader(object):
    '''
    批量图像读取


    '''
    dataset_dir = 'market1501'
    print('using market1501')

    def __init__(self, root='../data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')

        self._check_before_run()
        self.process_dir(self.train_dir,relabel=True)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        return img_paths

def process_dir(dir_path):

    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

    return img_paths

if __name__ == '__main__':
    # dr = data_reader()
    # dr.dataset_dir
    process_dir('../data/market1501/query')

    print(process_dir('../data/market1501/query'))

    embed()