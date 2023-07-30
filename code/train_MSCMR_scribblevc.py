import argparse
import importlib
import os
import random
from time import strftime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelAccuracy
from torchvision import transforms

from dataloaders.dataset_scribblevc import BaseDataSets, RandomGenerator, Zoom, ACDCDataSets, MSCMRDataSets
from tool import pyutils
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from utils.losses import pDLoss, SupConLoss
from val_2D_scribblevc import test_single_volume_CAM as test_single_volume, calculate_metric_percase


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--max_epoches", default=200, type=int)
    parser.add_argument("--network", default="network.scribbleVC", type=str)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float, help='optimizer weight decay')
    parser.add_argument("--arch", default='ACDC', type=str)
    parser.add_argument("--session_name", default="TransCAM", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--pretrain_weights", default='', type=str)
    parser.add_argument("--tblog", default='MSCMR/scribbleVC', type=str)

    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--root_path', type=str,
                        default='../data/MSCMR', help='Name of Experiment')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[256, 256],
                        help='patch size of network input')
    parser.add_argument('--fold', type=str,
                        default='fold1', help='cross validation')
    parser.add_argument('--sup_type', type=str,
                        default='scribble', help='supervision type')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--exp', type=str,
                        default='MSCMR/scribbleVC', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='scribbleVC', help='model_name')
    parser.add_argument('--optimizer', type=str,
                        default='adamw', help='optimizer name')
    parser.add_argument('--lrdecay', action="store_true", help='lr decay')
    parser.add_argument('--linear_layer', action="store_true", help='linear layer')
    parser.add_argument('--bilinear', action="store_false", help='use bilinear in Upsample layer')

    parser.add_argument('--weight_pseudo_loss', type=float, default=0.1, help='pseudo label loss')
    parser.add_argument('--weight_crf', type=float, default=0.1, help='crf loss')
    parser.add_argument('--weight_cls', type=float, default=0.1, help='cls loss')
    parser.add_argument('--temp', type=float, default=0.1, help='temperature for contrastive loss function SupConLoss')
    parser.add_argument('--no_class_rep', action="store_true", help='ban class representation')
    parser.add_argument("--val_every_epoches", default=1, type=int)
    parser.add_argument('--val_mode', action="store_true")
    parser.add_argument('--num_classes', default=4, type=int)


    args = parser.parse_args()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    start_time = strftime("%Y_%m_%d_%H_%M_%S")
    logdir = os.path.join(snapshot_path, "{}_log.txt".format(start_time))
    pyutils.Logger(logdir)
    print("log in ", logdir)
    print(vars(args))

    num_classes = args.num_classes
    model = getattr(importlib.import_module(args.network), 'scribbleVC_' + args.arch)(linear_layer=args.linear_layer,
                                                                               bilinear=args.bilinear,
                                                                               num_classes=num_classes,
                                                                               batch_size=args.batch_size)  # get Net_sm from network.conformer_CAM
    print('model is from', model.__class__)
    # print(model)

    tblogger = SummaryWriter(os.path.join("./tblog", "{}__{}".format(args.tblog, start_time), "train"))
    tblogger_valid = SummaryWriter(os.path.join("./tblog", "{}__{}".format(args.tblog, start_time), "valid"))

    # ----- Add from WSL4MIS -----
    db_train = MSCMRDataSets(base_dir=args.root_path, split="train", transform=
    # TwoCropTransform(
    transforms.Compose([RandomGenerator(args.patch_size)]), fold=args.fold, sup_type=args.sup_type)

    db_val = MSCMRDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    batch_size = args.batch_size
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    # loss definition
    if args.sup_type == "label":
        ce_loss = CrossEntropyLoss(ignore_index=0)
        dice_loss = pDLoss(num_classes, ignore_index=0)
    elif args.sup_type == "scribble":
        ce_loss = CrossEntropyLoss(ignore_index=4)
        dice_loss = pDLoss(num_classes, ignore_index=4)

    gatecrf_loss = ModelLossSemsegGatedCRF()
    loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    loss_gatedcrf_radius = 5

    cls_loss = BCEWithLogitsLoss()

    contrastive_loss = SupConLoss(temperature=args.temp)

    best_performance = 0.0
    best_epoch = 0
    iter_num = 0
    max_iterations = args.max_epoches * len(trainloader)

    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wt_dec, eps=1e-8)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_dec, eps=1e-8)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=0.0001)

    if len(args.pretrain_weights) != 0:
        print("Load pretrain weight from", args.pretrain_weights)
        model.load_state_dict(torch.load(args.pretrain_weights))
    model = model.cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_ce', 'loss_pseudo', 'loss_crf', 'loss_cls')
    train_dice = 0
    train_accuracy = MultilabelAccuracy(num_labels=num_classes-1).cuda()

    for ep in range(args.max_epoches):
        train_metric_list = []
        for iter, sampled_batch in enumerate(trainloader):
            img, label = sampled_batch['image'], sampled_batch['label']
            img, label = img.cuda(), label.cuda()
            category = sampled_batch['category'].cuda()

            pred1, pred2, cls_output = model(img, ep=ep, model_type = "train") \
                                            if not args.no_class_rep else model(img, 0)

            outputs_soft1 = torch.softmax(pred1, dim=1)
            outputs_soft2 = torch.softmax(pred2, dim=1)

            loss_ce1 = ce_loss(pred1, label[:].long())
            loss_ce2 = ce_loss(pred2, label[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2) if (label.unique() != 4).sum() else torch.tensor(0)
            loss = loss_ce


            beta = random.random() + 1e-10
            if args.weight_pseudo_loss:
                pseudo_supervision = torch.argmax(((torch.min(outputs_soft1.detach(), outputs_soft2.detach()) > 0.5) * \
                                                   (beta * outputs_soft1.detach() + (
                                                               1.0 - beta) * outputs_soft2.detach())),
                                                  dim=1, keepdim=False)  
                loss_pse_sup = 0.5 * (dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(1)) +
                                      dice_loss(outputs_soft2,
                                                pseudo_supervision.unsqueeze(1)))  

                loss = loss + args.weight_pseudo_loss * loss_pse_sup

            ensemble_pred = (beta * outputs_soft1 + (1.0 - beta) * outputs_soft2)

            if args.weight_crf:
                out_gatedcrf = gatecrf_loss(
                    ensemble_pred,
                    loss_gatedcrf_kernels_desc,
                    loss_gatedcrf_radius,
                    img,
                    args.patch_size[0],
                    args.patch_size[1],
                )["loss"]
                loss = loss + args.weight_crf * out_gatedcrf

            if args.weight_cls:
                loss_cls = sum([cls_loss(o, category.float()) / len(cls_output) for o in cls_output])
                loss = loss + args.weight_cls * loss_cls

                preds = 0.5 * cls_output[0] + 0.5 * cls_output[1]
                acc = train_accuracy(preds, category)

            if (ep + 1) % args.val_every_epoches == 0:
                out = torch.argmax(ensemble_pred.detach(), dim=1)
                prediction = out.cpu().detach().numpy()
                metric_i = []
                for i in range(1, num_classes):
                    metric_i.append(calculate_metric_percase(prediction == i, sampled_batch['gt'].cpu().detach().numpy() == i))
                train_metric_list.append(metric_i)

            if loss != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            iter_num = iter_num + 1
            if args.optimizer == 'SGD' or args.lrdecay:
                lr_ = args.lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                tblogger.add_scalar('info/lr', lr_, iter_num)

            avg_meter.add({'loss': loss.item(),
                           'loss_crf': out_gatedcrf.item() if args.weight_crf != 0 else 0,
                           'loss_cls': loss_cls.item() if args.weight_cls != 0 else 0})
            if loss_ce != 0:   avg_meter.add({'loss_ce': loss_ce.item()})
            if args.weight_pseudo_loss != 0:   avg_meter.add({'loss_pseudo': loss_pse_sup.item()})

        else:
            if not args.val_mode:
                print('epoch: %5d' % ep,
                      'loss: %.4f' % avg_meter.get('loss'), flush=True)
                tblogger.add_scalar('loss/loss', avg_meter.get('loss'), ep)
                tblogger.add_scalar('loss/loss_ce', avg_meter.get('loss_ce'), ep)
                if args.weight_pseudo_loss != 0:   tblogger.add_scalar('loss/loss_pseudo', avg_meter.get('loss_pseudo'), ep)
                if args.weight_crf != 0:   tblogger.add_scalar('loss/loss_crf', avg_meter.get('loss_crf'), ep)
                if args.weight_cls != 0:   tblogger.add_scalar('loss/loss_cls', avg_meter.get('loss_cls'), ep)

                if args.weight_cls:
                    total_train_acc = train_accuracy.compute()
                    print(f"train Accuracy on epoch {ep}: {total_train_acc}")
                    tblogger.add_scalar('metric/acc', total_train_acc, ep)
                    train_accuracy.reset()

            if (ep + 1) % args.val_every_epoches == 0:
                model.eval()
                if not args.val_mode:
                    train_metric_list = np.nanmean(np.array(train_metric_list),  axis=0)
                    train_dice = np.mean(train_metric_list, axis=0)[0]
                    mean_hd95 = np.mean(train_metric_list, axis=0)[1]
                    print('epoch %5d  train_dice : %.4f train_hd95 : %.4f' % (
                        ep, train_dice, np.mean(train_metric_list, axis=0)[1]), flush=True)
                    for class_i in range(num_classes - 1):
                        tblogger.add_scalar('metric/{}_dice'.format(class_i + 1),
                                            train_metric_list[class_i, 0], ep)
                        tblogger.add_scalar('hd95/{}_hd95'.format(class_i + 1),
                                            train_metric_list[class_i, 1], ep)
                    tblogger.add_scalar('metric/dice', train_dice, ep)
                    tblogger.add_scalar('hd95/hd95', mean_hd95, ep)

                metric_list = []
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,
                        patch_size=args.patch_size, epoch=ep,
                        model_type = 'val' if not args.no_class_rep else None)
                    metric_list.append(metric_i)

                metric_list = np.nanmean(np.array(metric_list), axis=0)
                for class_i in range(num_classes - 1):
                    tblogger_valid.add_scalar('metric/{}_dice'.format(class_i + 1),
                                        metric_list[class_i, 0], ep)
                    tblogger_valid.add_scalar('hd95/{}_hd95'.format(class_i + 1),
                                        metric_list[class_i, 1], ep)
                performance = np.nanmean(metric_list, axis=0)[0]
                mean_hd95 = np.nanmean(metric_list, axis=0)[1]
                tblogger_valid.add_scalar('metric/dice', performance, ep)
                tblogger_valid.add_scalar('hd95/hd95', mean_hd95, ep)

                if performance > 0.85:
                    print("Update high dice score model!")
                    file_name = os.path.join(snapshot_path, '{}_{}_model.pth'.format(args.model, str(performance)[0:6]))
                    torch.save(model.state_dict(), file_name)
                if (ep + 1) % 100 == 0:
                    print("{} model!".format(ep))
                    file_name = os.path.join(snapshot_path, '{}_{}_model.pth'.format(args.model, ep))
                    torch.save(model.state_dict(), file_name)
                if performance > best_performance:
                    best_performance = performance
                    best_epoch = ep
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))

                    try:
                        torch.save(model.module.state_dict(), save_best)
                    except AttributeError:
                        torch.save(model.state_dict(), save_best)
                    print('best model in epoch %5d  mean_dice : %.4f' % (ep, performance))

                print(
                    'epoch %5d  mean_dice : %.4f mean_hd95 : %.4f' % (ep, performance, mean_hd95), flush=True)
                model.train()

            avg_meter.pop()
    print('best model in epoch %5d  mean_dice : %.4f' % (best_epoch, best_performance))
    print('save best model in {}/{}_best_model.pth'.format(snapshot_path, args.model))
    try:
        torch.save(model.module.state_dict(), os.path.join(snapshot_path,
                                                           '{}_final_model.pth'.format(args.model)))
    except AttributeError:
        torch.save(model.state_dict(), os.path.join(snapshot_path,
                                                    '{}_final_model.pth'.format(args.model)))
