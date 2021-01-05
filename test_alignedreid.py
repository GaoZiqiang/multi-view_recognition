from __future__ import absolute_import
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import models
from util.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLossAlignedReID
from util import data_manager
from util import transforms as T
from util.dataset_loader import ImageDataset
from util.utils import Logger
from util.utils import AverageMeter, Logger, save_checkpoint
from util.eval_metrics import evaluate
from util.optimizers import init_optim
from util.samplers import RandomIdentitySampler
from IPython import embed

# matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed


# 作为C++调用Python的测试
def printHello():
    print('Hello World!')


def main():
    # 第四个参数：use_gpu，不需要显示的指定
    use_gpu = torch.cuda.is_available()
    # if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False

    dataset = data_manager.init_img_dataset(
        root='data', name='market1501', split_id=0,# 暂时存在问题
        # cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    # data augmentation
    transform_test = T.Compose([
        # T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # trainloader = DataLoader(
    #     ImageDataset(dataset.train, transform=transform_train),
    #     sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
    #     batch_size=args.train_batch, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=True,
    # )

    # 第二个参数：queryloader
    queryloader = DataLoader(
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

    # print("Initializing model: {}".format(args.arch))


    # 第一个参数：model
    model = models.init_model(name='resnet50', num_classes=dataset.num_train_pids, loss={'softmax', 'metric'},
                              aligned=True, use_gpu=use_gpu)


    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion_class = CrossEntropyLoss(use_gpu=use_gpu)
    criterion_metric = TripletLossAlignedReID(margin=0.3)
    optimizer = init_optim('adam', model.parameters(), 0.0002, 0.0005)


    scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    start_epoch = 0

    # if args.resume:
    #     print("Loading checkpoint from '{}'".format(args.resume))
    #     checkpoint = torch.load(args.resume)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()


    test(model, queryloader, galleryloader, use_gpu)

    print('------测试结束------')

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
    print('------start testing------')
    cmc1 = []
    cmc2 = []
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        # 计算query集的features
        qf, q_pids, q_camids, lqf = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)
            q_pids.extend(pids)
            q_camids.extend(camids)

        # embed()
        # 对tensor进行拼接,axis=0表示进行竖向拼接
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        # 计算gallery集的features
        gf, g_pids, g_camids, lgf = [], [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)
            g_pids.extend(pids)
            g_camids.extend(camids)

        # 打个断点，看一下gf
        # embed()
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, 32))

    # 下面这些是处理要点
    # feature normlization　特征标准化
    qf = 1. * qf / (torch.norm(qf, 2, dim=-1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
    # 这是啥
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())  # 矩阵相乘
    distmat = distmat.numpy()

    # 用于测试
    mm, nn = distmat.shape[0], distmat.shape[1]
    min = [1, 1, 1, 1, 1, 1, 1, 1]  # min数组的大小应该等于mm
    num = 0
    for i in range(mm):
        for j in range(nn):
            if distmat[i][j] < min[i]:
                min[i] = distmat[i][j]
        if min[i] < 0.4:
            num += 1
    print('经多视角识别后的person_num为:', num)

    # embed()
    # 下面对distmat进行处理，若distmat<xx则做某操作，否则，做某操作
    # embed()

    # 使用global特征
    print("Only using global branch")
    from util.distance import low_memory_local_dist
    lqf = lqf.permute(0, 2, 1)
    lgf = lgf.permute(0, 2, 1)
    # 计算local_distmat
    local_distmat = low_memory_local_dist(lqf.numpy(), lgf.numpy(), aligned=not False)


    print("Using global and local branches")
    # total distmat = local_distmat + distmat(global)
    distmat = local_distmat + distmat

    print("Computing CMC and mAP")
    # 打一个断点，对distmat进行排序
    # embed()
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)
    print("cms's shape: ", cmc.shape)
    print("cms's type: ", cmc.dtype)
    # embed()

    # embed()

    # cmc1 = []
    # print("cmc2's shape: ",cmc2.shape)
    print("------Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        cmc1.append(cmc[r - 1])
    print("------------------")

    # embed()

    # plt.plot(ranks,cmc1,label='ranking',color='red',marker='o',markersize=5)
    # plt.ylabel('Accuracy')
    # plt.xlabel('Rank_num')
    # plt.title('Result of Ranking and Re-ranking(query_tank_cam=5)')
    # plt.legend()
    # plt.show()
    #
    # 进行reranking
    # if args.reranking:
    #     from util.re_ranking import re_ranking
    #     if args.test_distance == 'global':
    #         print("Only using global branch for reranking")
    #         distmat = re_ranking(qf,gf,k1=20, k2=6, lambda_value=0.3)
    #     else:
    #         local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(),aligned= not args.unaligned)
    #         local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(),aligned= not args.unaligned)
    #         local_dist = np.concatenate(
    #             [np.concatenate([local_qq_distmat, local_distmat], axis=1),
    #              np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
    #             axis=0)
    #         if args.test_distance == 'local':
    #             print("Only using local branch for reranking")
    #             distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=True)
    #         elif args.test_distance == 'global_local':
    #             print("Using global and local branches for reranking")
    #             distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=False)
    #     print("Computing CMC and mAP for re_ranking")
    #     cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)
    #
    #     # cmc2 = []
    #     print("Results ----------")
    #     print("mAP(RK): {:.1%}".format(mAP))
    #     print("CMC curve(RK)")
    #     for r in ranks:
    #         print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    #         cmc2.append(cmc[r - 1])
    #     print("------------------")
    #
    #
    #
    # # matlpot
    # # print("----cmc2----",cmc2)
    # # print("cmc1's value------")
    # # print(cmc1)
    # plt.plot(ranks,cmc1,label='ranking',color='red',marker='o',markersize=5)
    # plt.plot(ranks,cmc2,label='re-ranking',color='blue',marker='o',markersize=5)
    # plt.ylabel('Accuracy')
    # plt.xlabel('Rank_num')
    # plt.title('Result of Ranking and Re-ranking(query_tank_cam=5)')
    # plt.legend()
    # plt.savefig('/home/gaoziqiang/tempt/tank_cam5.png')
    # plt.show()

    return cmc[0]


if __name__ == '__main__':
    main()
