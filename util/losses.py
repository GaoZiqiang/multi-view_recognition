from __future__ import absolute_import
# import aligned
# from aligned.local_dist import *
# from aligned.local_dist import *
# import aligned

import torch
from torch import nn

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['DeepSupervision', 'CrossEntropyLoss','CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss', 'RingLoss']


# 下面的几个ｄef都是因为无法导入aligned这个包而引起的
def hard_example_mining(dist_mat, labels, return_inds=False):
  """For each anchor, find the hardest positive and negative sample.
  Args:
    dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
    labels: pytorch LongTensor, with shape [N]
    return_inds: whether to return the indices. Save time if `False`(?)
  Returns:
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    p_inds: pytorch LongTensor, with shape [N];
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
  NOTE: Only consider the case in which all labels have same num of samples,
    thus we can cope with all anchors in parallel.
  """

  assert len(dist_mat.size()) == 2
  assert dist_mat.size(0) == dist_mat.size(1)
  N = dist_mat.size(0)

  # shape [N, N]
  is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
  is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

  # `dist_ap` means distance(anchor, positive)
  # both `dist_ap` and `relative_p_inds` with shape [N, 1]
  dist_ap, relative_p_inds = torch.max(
    dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
  # `dist_an` means distance(anchor, negative)
  # both `dist_an` and `relative_n_inds` with shape [N, 1]
  dist_an, relative_n_inds = torch.min(
    dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
  # shape [N]
  dist_ap = dist_ap.squeeze(1)
  dist_an = dist_an.squeeze(1)

  if return_inds:
    # shape [N, N]
    ind = (labels.new().resize_as_(labels)
           .copy_(torch.arange(0, N).long())
           .unsqueeze( 0).expand(N, N))
    # shape [N, 1]
    p_inds = torch.gather(
      ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    n_inds = torch.gather(
      ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    # shape [N]
    p_inds = p_inds.squeeze(1)
    n_inds = n_inds.squeeze(1)
    return dist_ap, dist_an, p_inds, n_inds

  return dist_ap, dist_an

def shortest_dist(dist_mat):
  """Parallel version.
  求最短路径
  Args:
    dist_mat: pytorch Variable, available shape:
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
  Returns:
    dist: three cases corresponding to `dist_mat`:
      1) scalar
      2) pytorch Variable, with shape [N]
      3) pytorch Variable, with shape [*]
  """
  m, n = dist_mat.size()[:2]
  # Just offering some reference for accessing intermediate distance.
  dist = [[0 for _ in range(n)] for _ in range(m)]
  for i in range(m):
    for j in range(n):
      if (i == 0) and (j == 0):
        dist[i][j] = dist_mat[i, j]
      elif (i == 0) and (j > 0):
        dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
      elif (i > 0) and (j == 0):
        dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
      else:
        # 等价表示：取上边和右边两个方向下来的路径的最小值
        dist[i][j] = torch.min(dist[i-1][j] + dist[i][j],dist[i][j-1] + dist[i][j])
        dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
  dist = dist[-1][-1]
  return dist

def batch_euclidean_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [Batch size, Local part, Feature channel]
    y: pytorch Variable, with shape [Batch size, Local part, Feature channel]
  Returns:
    dist: pytorch Variable, with shape [Batch size, Local part, Local part]
  """
  assert len(x.size()) == 3
  assert len(y.size()) == 3
  assert x.size(0) == y.size(0)
  assert x.size(-1) == y.size(-1)

  N, m, d = x.size()
  N, n, d = y.size()

  # shape [N, m, n]
  xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
  yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
  dist = xx + yy
  dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist


def batch_local_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [N, m, d]
    y: pytorch Variable, with shape [N, n, d]
  Returns:
    dist: pytorch Variable, with shape [N]
  """
  assert len(x.size()) == 3
  assert len(y.size()) == 3
  assert x.size(0) == y.size(0)
  assert x.size(-1) == y.size(-1)

  # shape [N, m, n]
  dist_mat = batch_euclidean_dist(x, y)
  dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
  # shape [N]
  dist = shortest_dist(dist_mat.permute(1, 2, 0))
  return dist

def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss

class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    """
    def __init__(self, use_gpu=True):
        super(CrossEntropyLoss, self).__init__()
        self.use_gpu = use_gpu
        self.crossentropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        if self.use_gpu: targets = targets.cuda()
        loss = self.crossentropy_loss(inputs, targets)
        return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        """
        定义要使用的参数
        Args:
            margin: 设置margin预值
            targets: ground truth labels with shape (num_classes)
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        # ranking_loss是pytorch的一个loss，其计算取决于四个参数ap,an,margin,y=1
        # ap:正样本距离，an:负样本距离
        # 损失计算公式 loss = Relu(ap - y*an + margin)
        # 在这里我们用ranking_loss作为我们的Triplet loss,事实上，pytorch已经有了自己的Triplet loss计算方法
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag
        # from IPython import embed
        # embed()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 这里的inputs是global features. feature matrix with shape (batch_size, feat_dim) 特征矩阵或者叫特征图矩阵，shape为batch_size*feat_dim，例如32*2018
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)# inputs.size()的第一个参数为batch_size，也就是一个batch的图像数量


        # 如下行所示，可以进行归一化处理
        # 数据归一化处理很重要，可以防止出现NAN问题
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)


        # Compute pairwise distance, replace by the official when merged
        # 对输入的特征矩阵求平方，然后对各元素求和，最后拓展成为n*n的方阵
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist.t():对dist求转置，实为b^2
        # dist = dist + dist.t():a^2+b^2
        dist = dist + dist.t()
        # dist.addmm(a1,a2,a,b) = a1*dist + a2*a*b
        # 最终得到距离的平方方阵dist
        dist.addmm(1, -2, inputs, inputs.t())

        # 开方，得到距离方阵dist
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        # torch.eq()函数得到一个n*n的0/1布尔矩阵
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # positive anchor和negative anchor
        dist_ap, dist_an = [], []
        for i in range(n):
            # 把距离矩阵中mask为1的值的最大值拿出来作为ap
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            # 把距离矩阵中mast为0的值的最大值拿出来作为an
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        # from IPython import embed
        # embed()

        # 得到ap、an的距离
        dist_ap = torch.cat(dist_ap)# 一个32的list
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss

        y = torch.ones_like(dist_ap)
        # 计算损失
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # if self.mutual:
        #     return loss, dist

        from IPython import embed
        embed()
        return loss

# 计算局部特征的损失
class TripletLossAlignedReID(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLossAlignedReID, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets, local_features):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
            local_features:local feature matrix
        """
        n = inputs.size(0)

        # 如下行所示，可以进行归一化处理
        # 数据归一化处理很重要，可以防止出现NAN问题
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        dist_ap,dist_an,p_inds,n_inds = hard_example_mining(dist,targets,return_inds=True)
        p_inds = p_inds.long()
        n_inds = n_inds.long()

        # from IPython import embed
        # embed()
        local_features = local_features.permute(0,2,1)

        p_local_features = local_features[p_inds]
        n_local_features = local_features[n_inds]
        local_dist_ap = batch_local_dist(local_features, p_local_features)
        local_dist_an = batch_local_dist(local_features, n_local_features)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        global_loss = self.ranking_loss(dist_an, dist_ap, y)
        local_loss = self.ranking_loss_local(local_dist_an,local_dist_ap, y)
        if self.mutual:
            return global_loss+local_loss,dist
        return global_loss,local_loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

class RingLoss(nn.Module):
    """Ring loss.
    
    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """
    def __init__(self, weight_ring=1.):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))
        self.weight_ring = weight_ring

    def forward(self, x):
        l = ((x.norm(p=2, dim=1) - self.radius)**2).mean()
        return l * self.weight_ring

class KLMutualLoss(nn.Module):
    def __init__(self):
        super(KLMutualLoss,self).__init__()
        self.kl_loss = nn.KLDivLoss(size_average=False)
        self.log_softmax = nn.functional.log_softmax
        self.softmax = nn.functional.softmax
    def forward(self, pred1, pred2):
        pred1 = self.log_softmax(pred1, dim=1)
        pred2 = self.softmax(pred2, dim=1)
        #loss = self.kl_loss(pred1, torch.autograd.Variable(pred2.data))
        loss = self.kl_loss(pred1, pred2.detach())
        # from IPython import embed
        # embed()
        #print(loss)
        return loss

class MetricMutualLoss(nn.Module):
    def __init__(self):
        super(MetricMutualLoss, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, dist1, dist2,pids):
        loss = self.l2_loss(dist1, dist2)
        # from IPython import embed
        # embed()
        print(loss)
        return loss

# 测试验证
if __name__ == '__main__':
    target = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8]
    target = torch.Tensor(target)# 转换为张量Tensor
    features = torch.Tensor(32,2048)
    local_features = torch.randn(32,128,8)
    # 实例化两个类的实例
    a = TripletLoss()
    b = TripletLossAlignedReID()
    glocal_loss,local_loss = b(features,target,local_features)
    # print('-----打印a.forward(features,target)----')
    # print(a.forward(features,target))
    from IPython import embed
    embed()



if __name__ == '__main__':
    target = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8]
    target = torch.Tensor(target)# 转换为张量Tensor
    features = torch.Tensor(32,2048)
    # local_features = torch.randn(32,128,8)
    # 实例化两个类的实例
    a = TripletLoss()
    # b = TripletLossAlignedReID()
    glocal_loss = a(features,target)
    # print('-----打印a.forward(features,target)----')
    # print(a.forward(features,target))
    from IPython import embed
    embed()