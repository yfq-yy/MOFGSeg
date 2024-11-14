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
from train_baseline_sup import evaluatemask
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from model.model_helper import ModelBuilder
from modulelist import *

parser = argparse.ArgumentParser(description='Semi-Supervised Learning for Multi-Object Segmentation with  Fine-Grained Classes')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

    cudnn.enabled = True
    cudnn.benchmark = True

    model = ModelBuilder(cfg['model'])
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    ema = EMA(model, 0.99) 
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    ema.model.cuda()
     # Mean teacher model

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    previous_best = 0.0
    best_epoch = 0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])
        ema.model.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        previous_best = checkpoint['previous_best']

    logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f} in Epoch {:}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best, best_epoch))

    #model.module.decoder.set_SMem_status(epoch=epoch, isVal=True)
    #yfq
    dataset = 'pascal'
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class = evaluatemask(model, valloader, eval_mode, cfg, dataset)

    
    for (cls_idx, iou) in enumerate(iou_class):
        logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))



if __name__ == '__main__':
    main()
