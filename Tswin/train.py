# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------


import datetime
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from build import build_model
from config import get_config
from logger import create_logger
from losses import cal_selfsupervised_loss
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from sklearn.metrics import classification_report
from timm.utils import AverageMeter, accuracy
from torch.utils.tensorboard import SummaryWriter

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "5678"


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def _weight_decay(init_weight, epoch, warmup_epochs=10, total_epoch=300):
    if epoch <= warmup_epochs:  # noqa: SIM108
        cur_weight = min(init_weight / warmup_epochs * epoch, init_weight)
    else:
        cur_weight = init_weight * (1.0 - (epoch - warmup_epochs) / (total_epoch - warmup_epochs))
    return cur_weight


def main(data_loader_train, data_loader_val, num_classes, save_path="./best-modelforall.pth"):
    tb_write = SummaryWriter()
    config = get_config()

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # # gradient accumulation also need to scale the learning rate
    # if config.TRAIN.ACCUMULATION_STEPS > 1:
    #     linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    # config.defrost()
    # config.TRAIN.BASE_LR = linear_scaled_lr
    # config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    # config.TRAIN.BASE_LR = 2e-3
    # config.TRAIN.WARMUP_LR = 1e-3
    # config.TRAIN.MIN_LR = 1e-5
    # config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, num_classes)

    model.cuda()

    optimizer = build_optimizer(config, model)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # # supervised criterion
    # if config.AUG.MIXUP > 0.0:
    #     # smoothing is handled with mixup label transform
    #     criterion_sup = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.0:
    #     criterion_sup = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    criterion_sup = torch.nn.CrossEntropyLoss()

    # self-supervised criterion自监督准则
    criterion_ssup = cal_selfsupervised_loss

    max_accuracy = 0.0

    logger.info("Start training")
    start_time = time.time()

    init_lambda_drloc = 0.0
    for epoch in range(config.TRAIN.EPOCHS):
        if config.TRAIN.USE_DRLOC:
            init_lambda_drloc = _weight_decay(config.TRAIN.LAMBDA_DRLOC, epoch, config.TRAIN.SSL_WARMUP_EPOCHS, config.TRAIN.EPOCHS)

        train_metrics = train_one_epoch(config, model, criterion_sup, criterion_ssup, data_loader_train, optimizer, epoch, lr_scheduler, logger, init_lambda_drloc)
        abc = OrderedDict(epoch=epoch)
        abc.update([("train_" + k, v) for k, v in train_metrics.items()])

        acc1, acc5, loss = validate(data_loader_val, model, logger)
        tags = ["val_acc1", "val_loss", "train_loss", "learn_rate"]
        tb_write.add_scalar(tags[0], acc1, epoch)
        tb_write.add_scalar(tags[1], loss, epoch)
        tb_write.add_scalar(tags[2], abc["train_loss"], epoch)
        # tb_write.add_scalar(tags[1],)
        tb_write.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        logger.info(f"Accuracy of the network on the test images: {acc1:.1f}%")
        if acc1 > max_accuracy:
            torch.save(model.state_dict(), save_path)
            logger.info(f"{save_path} saved !!!")

            # save_checkpoint_best(config, epoch, model_without_ddp, max_accuracy, save_path, logger)
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    return model


def train_one_epoch(config, model, criterion_sup, criterion_ssup, data_loader, optimizer, epoch, lr_scheduler, logger, lambda_drloc):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    cc = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
    init.uniform_(cc, a=0, b=1)
    for idx, (samples, targets) in enumerate(data_loader):
        # print("SHAPE -----------", samples.shape, targets.shape)
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion_sup(outputs.sup, targets)
            if config.TRAIN.USE_DRLOC:
                loss_ssup, ssup_items, d = criterion_ssup(outputs, config, lambda_drloc)
                loss = (1 - cc.item()) * loss + loss_ssup * cc.item()
                loss += loss_ssup * cc.item()
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())

            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion_sup(outputs.sup, targets)
            if config.TRAIN.USE_DRLOC:
                loss_ssup, ssup_items = criterion_ssup(outputs, config, lambda_drloc)
                loss = (1 - cc.item()) * loss + loss_ssup * cc.item()
                # loss += loss_ssup*cc.item()
                # print(cc.item())
                # loss = (loss-1.0).abs()+1.0
            optimizer.zero_grad()

            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())

            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        # if idx % config.PRINT_FREQ == 0:
    lr = optimizer.param_groups[0]["lr"]
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    etas = batch_time.avg * (num_steps - idx)
    logger.info(
        # f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
        f"Train: [{epoch}/{config.TRAIN.EPOCHS}]\t"
        f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
        f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
        f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
        f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
        f"mem {memory_used:.0f}MB"
    )
    if config.TRAIN.USE_DRLOC:
        logger.info(f"weights: drloc {lambda_drloc:.4f}")
        logger.info(" ".join(["%s: [%.4f]" % (key, value) for key, value in ssup_items.items()]))

    epoch_time = time.time() - start
    # print(cc.item())

    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return OrderedDict([("loss", loss_meter.avg)])


# @torch.no_grad()
# def validate(config, data_loader, model, logger):
#     criterion = torch.nn.CrossEntropyLoss()
#     model.eval()

#     batch_time = AverageMeter()
#     loss_meter = AverageMeter()
#     acc1_meter = AverageMeter()
#     acc5_meter = AverageMeter()

#     end = time.time()
#     for idx, (images, target) in enumerate(data_loader):
#         images = images.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)

#         # compute output
#         output = model(images)

#         # measure accuracy and record loss
#         loss = criterion(output.sup, target)
#         acc1, acc5 = accuracy(output.sup, target, topk=(1, 5))

#         loss_meter.update(loss.item(), target.size(0))
#         acc1_meter.update(acc1.item(), target.size(0))
#         acc5_meter.update(acc5.item(), target.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         # if idx % config.PRINT_FREQ == 0:
#     memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#     logger.info(
#         # f"Test: [{idx}/{len(data_loader)}]\t"
#         f"TEST:\t"
#         f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
#         f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
#         f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
#         f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
#         f"Mem {memory_used:.0f}MB"
#     )
#     logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")
#     return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def validate(data_loader, model, logger):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    all_preds = []
    all_labels = []

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output.sup, target)
        acc1, acc5 = accuracy(output.sup, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # collect predictions and targets for classification report
        preds = torch.argmax(output.sup, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(target.cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

    # flatten all_preds and all_labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f"TEST:\t"
        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
        f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
        f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
        f"Mem {memory_used:.0f}MB"
    )
    logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")

    # ADD: classification report
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == "__main__":
    main()
