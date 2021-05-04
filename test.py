from __future__ import absolute_import

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import models
from util.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLossAlignedReID
from util import data_manager# 此为我们的data_manager工具类
from util import transforms as T
from util.dataset_loader import ImageDataset
from util.utils import AverageMeter, Logger, save_checkpoint
from util.optimizers import init_optim


def main():
    # 第四个参数：use_gpu，不需要显示的指定
    use_gpu = torch.cuda.is_available()
    # if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False

    # 其实可以换一种写法
    dataset = data_manager.Market1501(root='data')

    # data augmentation
    transform_test = T.Compose([
        # T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 第二个参数：queryloader
    queryloader = DataLoader(
        # 问题：dataset.query哪里来的？ 答：来自dataset = data_manager.Market1501(root='data')
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=32, shuffle=False, num_workers=4,
        pin_memory=pin_memory, drop_last=False,
    )
    # 第三个参数：galleryloader
    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=32, shuffle=False, num_workers=4,
        pin_memory=pin_memory, drop_last=False,
    )

    model = models.init_model(name='resnet50', num_classes=8, loss={'softmax', 'metric'},
                              aligned=True, use_gpu=use_gpu)

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion_class = CrossEntropyLoss(use_gpu=use_gpu)
    criterion_metric = TripletLossAlignedReID(margin=0.3)
    optimizer = init_optim('adam', model.parameters(), 0.0002, 0.0005)


    scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # embed()
    test(model, queryloader, galleryloader, use_gpu)

    return 0

def train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    xent_losses = AverageMeter()
    global_losses = AverageMeter()
    local_losses = AverageMeter()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)
        outputs, features, local_features = model(imgs)
        if args.htri_only:
            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, features, pids, local_features)
            else:
                global_loss, local_loss = criterion_metric(features, pids, local_features)
        else:
            if isinstance(outputs, tuple):
                xent_loss = DeepSupervision(criterion_class, outputs, pids)
            else:
                xent_loss = criterion_class(outputs, pids)

            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, features, pids, local_features)
            else:
                global_loss, local_loss = criterion_metric(features, pids, local_features)
        loss = xent_loss + global_loss + local_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), pids.size(0))
        xent_losses.update(xent_loss.item(), pids.size(0))
        global_losses.update(global_loss.item(), pids.size(0))
        local_losses.update(local_loss.item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLoss {xent_loss.val:.4f} ({xent_loss.avg:.4f})\t'
                  'GLoss {global_loss.val:.4f} ({global_loss.avg:.4f})\t'
                  'LLoss {local_loss.val:.4f} ({local_loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time, data_time=data_time,
                loss=losses, xent_loss=xent_losses, global_loss=global_losses, local_loss=local_losses))


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 8]):
    print('------开始测试------')
    # batch_time = AverageMeter()
    model.eval()

    with torch.no_grad():
        ### 计算query集的features
        qf,lqf = [], []# qf:query feature lqf:local query feature
        i = 0
        for batch_idx, imgs in enumerate(queryloader):
            i += 1
            if use_gpu: imgs = imgs.cuda()

            # end = time.time()
            # 使用model对图像进行特征提取
            features, local_features = model(imgs)
            # batch_time.update(time.time() - end)

            features = features.data.cpu()# 使用cpu进行处理
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)

        # 对tensor进行拼接,axis=0表示进行竖向拼接
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf, 0)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        ### 计算gallery集的features
        gf,lgf = [], []

        # end = time.time()
        for batch_idx, imgs in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            # end = time.time()
            # 使用resnet50进行图像特征提取
            features, local_features = model(imgs)
            # batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)

        # 打个断点，看一下gf
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf, 0)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    # print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, 32))

    # 下面这些是处理要点
    # feature normlization　特征标准化
    qf = 1. * qf / (torch.norm(qf, 2, dim=-1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
    # 矩阵的行列数
    m, n = qf.size(0), gf.size(0)

    torch.pow(qf, 2).sum(dim=1, keepdim=True)

    # 计算全局距离矩阵global distmat
    # torch.pow(qf,2)：求矩阵中各元素的平方
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())  # 矩阵相乘
    distmat = distmat.numpy()

    # 使用local_distmat
    from util.distance import low_memory_local_dist
    lqf = lqf.permute(0, 2, 1)
    lgf = lgf.permute(0, 2, 1)
    local_distmat = low_memory_local_dist(lqf.numpy(), lgf.numpy(), aligned=True)

    # Using global and local branches
    distmat = local_distmat + distmat
    # 归一化
    distmat = distmat / 10

    # 用于测试
    mm, nn = distmat.shape[0], distmat.shape[1]
    min = [1, 1, 1, 1, 1, 1, 1, 1]  # min数组的大小应该等于mm
    num = 0
    for i in range(mm):
        for j in range(nn):
            if distmat[i][j] < min[i]:
                min[i] = distmat[i][j]
        # 这里的判定两object是否为同一object的distance阈值为经验值，还需进一步优化
        if min[i] < 3.5:
            # print("min[i] is",min[i])
            num += 1
    print('各图像之间的相似度为：\n',1 - distmat)
    print('经多视角识别后的person_num为:', num)
    print('------测试结束------')

if __name__ == '__main__':
    main()
