import argparse
import logging
import os
import pprint
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from dataset.semi import SemiDataset
from train_baseline_sup import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter, batch_transform, generate_unsup_data
from util.dist_helper import setup_distributed
from model.model_helper import ModelBuilder
from modulelist import *

parser = argparse.ArgumentParser(description='Semi-Supervised Learning for Multi-Object Segmentation with  Fine-Grained Classes')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--strong_threshold', default=0.97, type=float)
parser.add_argument('--apply_reco', action='store_true', default=True)
parser.add_argument('--apply_aug', default='cutout', type=str, help='apply semi-supervised method: cutout cutmix classmix')
parser.add_argument('--weak_threshold', default=0.7, type=float) #yfq
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = ModelBuilder(cfg['model'])
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    ema = EMA(model, 0.99)  # Mean teacher model

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    ema.model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], 
                                                      output_device=local_rank, find_unused_parameters=False)


    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(**cfg['criterion_u']['kwargs']).cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False,sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    best_epoch = 0
    epoch = -1

    # laplace = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), dtype=np.int64)
    # laplace = laplace[np.newaxis, np.newaxis, ...]
    # laplace = torch.Tensor(laplace).cuda()

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        ema.model.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f} in Epoch {:}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best, best_epoch))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_edge = AverageMeter()
        total_loss_u = AverageMeter()
        #total_loss_u_f = AverageMeter()
        total_reco_loss = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        model.module.decoder.set_SMem_status(epoch=epoch, isVal=False)
        #model.decoder.set_SMem_status(epoch=epoch, isVal=False)
       
        for i, ((img_x, mask_x, edge_x), img_u_s) in enumerate(loader):

            img_x, mask_x, edge_x = img_x.cuda(), mask_x.cuda(), edge_x.cuda()
            mask_x[mask_x==255] = -1
            edge_x[edge_x==255] = 1 #yfq
            img_u_s = img_u_s.cuda()
        
          
            with torch.no_grad():
                ema.model.eval()
                pred_u, _, _ = ema.model(img_u_s)
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u, dim=1), dim=1)
                model.module.decoder.set_pseudo_prob_map(pred_u)
                model.module.decoder.set_pseudo_label(pseudo_labels)
        
                # random scale images first
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(img_u_s, pseudo_labels, pseudo_logits,
                                (cfg['crop_size'],cfg['crop_size']), (0.5, 2.0), apply_augmentation=False)

                # apply mixing strategy: cutout, cutmix or classmix
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=args.apply_aug)
        
                # apply augmentation: color jitter + flip + gaussian blur
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
                                (cfg['crop_size'],cfg['crop_size']), (1.0, 1.0), apply_augmentation=True)
            
             # generate labelled and unlabelled data loss
            pred_l, rep_l, edge_l = model(img_x)

            #pred_u0, _, _ = model(train_u_aug_data)
            pred_u, rep_u, _ = model(train_u_aug_data)
            rep_all = torch.cat((rep_l, rep_u))
            pred_all = torch.cat((pred_l, pred_u))
            # supervised-learning loss
            sup_loss = compute_supervised_loss(pred_l, mask_x)
            # edge-learning loss
            sup_edge_loss = dice_loss(edge_l, edge_x)
            # unsupervised-learning loss
            #unsup_loss_f = compute_unsupervised_loss(pred_u_large0, train_u_aug_label, train_u_aug_logits, args.strong_threshold)
            # unsupervised-learning loss
            unsup_loss = compute_unsupervised_loss(pred_u, train_u_aug_label, train_u_aug_logits, args.strong_threshold) #-strong_threshold', default=0.97, type=float)
            #unsup_loss = compute_supervised_loss(pred_u_large, pseudo_labels)
        # apply regional contrastive loss
            if args.apply_reco:
                with torch.no_grad():
                    train_u_aug_mask = train_u_aug_logits.ge(args.weak_threshold).float() #yfq
                    mask_all = torch.cat(((mask_x.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                    mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                    label_l = F.interpolate(label_onehot(mask_x, cfg['nclass']), size=pred_all.shape[2:], mode='nearest')
                    label_u = F.interpolate(label_onehot(train_u_aug_label, cfg['nclass']), size=pred_all.shape[2:], mode='nearest')
                    label_all = torch.cat((label_l, label_u))

                    prob_l = torch.softmax(pred_l, dim=1)
                    prob_u = torch.softmax(pred_u, dim=1)
                    prob_all = torch.cat((prob_l, prob_u))

                reco_loss = compute_reco_loss(rep_all, label_all, mask_all, prob_all, args.strong_threshold,
                                          args.temp, args.num_queries, args.num_negatives)
            else:
                reco_loss = torch.tensor(0.0)

            model.train()
            ema.model.train()
            loss = sup_loss + unsup_loss + sup_edge_loss+ reco_loss
            torch.distributed.barrier()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)

            total_loss.update(loss.item())
            total_loss_x.update(sup_loss.item())
            total_loss_u.update(unsup_loss.item())
            total_loss_edge.update(sup_edge_loss.item())
            total_reco_loss.update(reco_loss.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', sup_loss.item(), iters)
                writer.add_scalar('train/loss_u', unsup_loss.item(), iters)
                writer.add_scalar('train/loss_edge', sup_edge_loss.item(), iters)
                writer.add_scalar('train/reco_loss', reco_loss.item(), iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss edge: {:.3f}, Loss u: {:.3f}, reco loss: {:.3f}'
                            .format(i, total_loss.avg, total_loss_x.avg, total_loss_edge.avg, total_loss_u.avg, total_reco_loss.avg))
        model.decoder.set_SMem_status(epoch=epoch, isVal=True)
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if is_best:
            best_epoch = epoch
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': ema.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()