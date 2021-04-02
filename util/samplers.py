from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler
from IPython import embed
'''
图像处理类，对每个pid取四张不同序列号的图像，形成列表res[]

'''
class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    # param@data_source:即为dataset
    # param@num_instances:每个pid对应的图像张数
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        # index_dic是个字典，key为pid，value为该pid对应的图像的序列号（即index）
        self.index_dic = defaultdict(list)#　收集图像的序列号index（顺序号）放在index_dic
        # embed()
        for index, (_, pid, _) in enumerate(data_source):
            embed()
            self.index_dic[pid].append(index)
        # 将pid整合到一个list中
        self.pids = list(self.index_dic.keys())
        # 计算所有的id总数
        self.num_identities = len(self.pids)

        # from IPython import embed
        # embed()

    # 迭代器　返回　一个epoch
    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        # 使用ret[]列表来存放图像的序列号，每四个元素代表一张图像的序列号，多的图像会丢掉
        ret = []
        for i in indices:
            pid = self.pids[i]
            # index_dic[pid]：index_dic是个字典，key为pid，value为该pid对应的图像的序列号（即index），一个pid可能对应好多张图像
            t = self.index_dic[pid]
            # param@replace：当图像数量少于num_instances时，需进行重复其中的某些图像
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)

        from IPython import embed
        embed()

        return iter(ret)

    def __len__(self):
        # param@num_instances:每个pid对应的图像张数
        # param@num_identities = len(self.pids)
        return self.num_identities * self.num_instances


# 测试
if __name__ == '__main__':
    from data_manager import Market1501
    dataset = Market1501(root='/home/gaoziqiang/project/reid/deep-person-reid/data')
    sampler = RandomIdentitySampler(dataset.train,num_instances=4)
    a= sampler.__iter__()
