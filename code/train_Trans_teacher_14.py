import argparse
import logging
import os
import random
import shutil
import sys
import time
from itertools import cycle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms,ops
from torchvision.utils import make_grid
from tqdm import tqdm
import datetime
from dataloaders import utils
from dataloaders.dataset_semi import (BaseDataSets, RandomGenerator,TwoStreamBatchSampler)
from networks.discriminator import FCDiscriminator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume2
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from torch.nn import CosineSimilarity
from torch.utils.data.distributed import DistributedSampler
import math


"""选择GPU ID"""
# gpu_list = [1,2] #[0,1]
# gpu_list_str = ','.join(map(str, gpu_list))
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils.gate_crf_loss import ModelLossSemsegGatedCRF

parser = argparse.ArgumentParser()
parser.add_argument('--optim_name', type=str,default='adam', help='optimizer name')
parser.add_argument('--lr_scheduler', type=str,default='warmupCosine', help='lr scheduler') 

parser.add_argument('--root_path', type=str,
                    default='/mnt/sdd/tb/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC_Semi/Mean_Teacher', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_new', help='model_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=40,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=42, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=20,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=4,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--ema_decay2', type=float,  default=0.8, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.5, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

#trans parameters
parser.add_argument(
    '--cfg', type=str, default="/mnt/sdd/tb/WSL4MIS/code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

parser.add_argument('--my_lambda', type=float,  default=1, help='balance factor to control contrastive loss')
parser.add_argument('--tau', type=float,  default=1, help='temperature of the contrastive loss')

parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 2), type=int)
parser.add_argument("--kd_weights", type=int, default=15)

args = parser.parse_args()
config = get_config(args)
# 
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):


    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def create_model(ema=False):
        # Network definition
        # model = net_factory(net_type=args.model, in_chns=1,class_num=num_classes)
        model = ViT_seg(config, img_size=args.patch_size,num_classes=args.num_classes)      

        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    

    model = create_model()
    ema_model = create_model(ema=True)


    model=model.to(device) 
    ema_model =ema_model.to(device)

    num_gpus = torch.cuda.device_count()
    
    db_train_labeled = BaseDataSets(base_dir=args.root_path, num=8, labeled_type="labeled", fold=args.fold, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)]),sup_type=args.sup_type)
    db_train_unlabeled = BaseDataSets(base_dir=args.root_path, num=8, labeled_type="unlabeled", fold=args.fold, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)]))



    trainloader_labeled = DataLoader(db_train_labeled, batch_size=args.batch_size//2, shuffle=True,
                                     num_workers=16, pin_memory=True, drop_last=True,worker_init_fn=worker_init_fn)
    trainloader_unlabeled = DataLoader(db_train_unlabeled, batch_size=args.batch_size//2, shuffle=True,
                                       num_workers=16, pin_memory=True, drop_last=True,worker_init_fn=worker_init_fn)

    db_val = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="val", )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()
    # optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001) 
    max_epoch = max_iterations // len(trainloader_labeled) + 1
    warm_up_epochs = int(max_epoch * 0.1)
    if args.optim_name=='adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    elif args.optim_name=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,weight_decay=0.0001)
    elif args.optim_name=='adamW':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)    
    # elif args.optim_name=='Radam':
    #     optimizer = optim2.RAdam(model.parameters(), lr=base_lr, weight_decay=0.0001)   
        
        # warm_up_with_multistep_lr
    if args.lr_scheduler=='warmupMultistep':
        lr1,lr2,lr3 = int(max_epoch*0.25) , int(max_epoch*0.4) , int(max_epoch*0.6)
        lr_milestones = [lr1,lr2,lr3]
        # lr1,lr2,lr3,lr4 = int(max_epoch*0.15) , int(max_epoch*0.35) , int(max_epoch*0.55) , int(max_epoch*0.7)
        # lr_milestones = [lr1,lr2,lr3,lr4]
        warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
                                                else 0.1**len([m for m in lr_milestones if m <= epoch])
        scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = warm_up_with_multistep_lr)
    elif args.lr_scheduler=='warmupCosine':
        # warm_up_with_cosine_lr
        warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
                                else 0.5 * ( math.cos((epoch - warm_up_epochs) /(max_epoch - warm_up_epochs) * math.pi) + 1)
        scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = warm_up_with_cosine_lr)
    elif args.lr_scheduler=='autoReduce':
        scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=6, verbose=True, cooldown=2,min_lr=0)


    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)
    cos_sim = CosineSimilarity(dim=1,eps=1e-6)
    affinityenergyLoss=losses.SegformerAffinityEnergyLoss()
    criterion = torch.nn.MSELoss()


    gatecrf_loss = ModelLossSemsegGatedCRF()
    loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    loss_gatedcrf_radius = 5


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader_labeled)))
    lr_curve = list()
    iter_num = 0

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # train_sampler_labeled.set_epoch(epoch_num)
        for i, data in enumerate(zip(cycle(trainloader_labeled), trainloader_unlabeled)):
            sampled_batch_labeled, sampled_batch_unlabeled = data[0], data[1]

            volume_batch, label_batch = sampled_batch_labeled['image'], sampled_batch_labeled['label']
            label_batch_wr = sampled_batch_labeled['random_walker']
            crop_images = sampled_batch_labeled['crop_images']    
            boxes = sampled_batch_labeled['boxes']

            crop_images = crop_images.to(device)
            label_batch_wr = label_batch_wr.to(device)
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            unlabeled_volume_batch = sampled_batch_unlabeled['image'].to(device)


            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            ema_inputs = torch.cat([volume_batch,ema_inputs],0)

            volume_batch=torch.cat([volume_batch,unlabeled_volume_batch],0)



            outputs,attpred,att,out_feats =model(volume_batch) 
            # outputs_unlabeled,_,_,_= model(unlabeled_volume_batch)  

            outputs_unlabeled_soft = torch.softmax(outputs[args.labeled_bs:,...], dim=1)
            outputs_seg_soft = torch.softmax(outputs[:args.labeled_bs,...], dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs,...], label_batch[:].long())


            loss_ce_wr = ce_loss(outputs[:args.labeled_bs,...], label_batch_wr[:].long())
            loss_dice_wr= dice_loss(outputs_seg_soft, label_batch_wr.unsqueeze(1))
            #dice_loss(outputs_soft[:args.labeled_bs,...], label_batch.unsqueeze(1))
            # supervised_loss = 0.5 * (loss_dice + loss_ce)
            supervised_loss=loss_ce + 0.5 * (loss_ce_wr + loss_dice_wr)
            # loss=loss_ce
            # with torch.no_grad():
            #     ema_output,ema_attpred,_,_,calss_ema = ema_model(ema_inputs)
                
            #     ema_outputs_unlabeled_soft = torch.softmax(ema_output[args.labeled_bs:,...], dim=1)
            #     ema_outputs_seg_soft = torch.softmax(ema_output[:args.labeled_bs,...], dim=1)  
            # 
            #               
            out_seg_mlp=F.interpolate(out_feats[0], size=volume_batch.shape[2:], mode='bilinear', align_corners=False)                
        #     #consistency loss
            consistency_loss  = torch.mean((outputs_unlabeled_soft - out_seg_mlp[args.labeled_bs:,...]) ** 2)        
            consistency_weight = get_current_consistency_weight(iter_num // 300)


            # if iter_num < 1000:
            #     consistency_loss = 0.0
            # else:
            #     consistency_loss = torch.mean((outputs_unlabeled_soft - ema_outputs_unlabeled_soft) ** 2)
        #     with torch.cuda.amp.autocast():
        #         # -2: padded pixels;  -1: unlabeled pixels (其中60%-70%是没有标注信息的)
            unlabeled_RoIs = (label_batch == 4)
            unlabeled_RoIs=unlabeled_RoIs.type(torch.FloatTensor).to(device)
            label_batch[label_batch < 0] = 0                    
         #aff_loss
            outs = []        
            outs.append(out_feats[0][:args.labeled_bs,...])            
            outs.append(out_feats[1][:args.labeled_bs,...])
            outs.append(out_feats[2][:args.labeled_bs,...])
            outs.append(out_feats[3][:args.labeled_bs,...])


            affinityenergyloss = affinityenergyLoss(outs, att, unlabeled_RoIs,label_batch)
            affinity_loss = losses.get_aff_loss(attpred[:args.labeled_bs,...],label_batch_wr)


            # cosine similarity loss



            loss = 5*supervised_loss+affinityenergyloss*args.kd_weights+affinity_loss+consistency_weight*consistency_loss 
            #+affinity_loss#+affinityenergyLoss*args.kd_weights
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_
            ##更新学习率
            scheduler_lr.step()
            lr_iter = optimizer.param_groups[0]['lr']
            lr_curve.append(lr_iter)


            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_iter, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_ce, iter_num)
            writer.add_scalar('info/consistency_loss',consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume2(
                        sampled_batch["image"].to(device), sampled_batch["label"].to(device), model, device=device,classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

def backup_code(base_dir):
    ###备份当前train代码文件及dataset代码文件
    code_path = os.path.join(base_dir, 'code') 
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    train_name = os.path.basename(__file__)
    dataset_name = 'dataset_semi.py'
    # dataset_name2 = 'dataset_semi_weak_newnew_20.py'
    net_name1 = 'mix_transformer.py'  
    net_name2 = 'net_factory.py'  
    net_name3 = 'vision_transformer.py'
    net_name4 = 'head.py'

    shutil.copy('networks/' + net_name1, code_path + '/' + net_name1)
    shutil.copy('networks/' + net_name2, code_path + '/' + net_name2)
    shutil.copy('networks/' + net_name3, code_path + '/' + net_name3)
    shutil.copy('networks/' + net_name4, code_path + '/' + net_name4)
    shutil.copy('dataloaders/' + dataset_name, code_path + '/' + dataset_name)
    shutil.copy(train_name, code_path + '/' + train_name)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/mnt/sdd/tb/work_dirs/model_/{}_{}/{}-{}".format(args.exp, args.fold, args.sup_type,datetime.datetime.now())
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    backup_code(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
