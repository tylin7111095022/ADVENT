# --------------------------------------------------------
# Domain adpatation training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from libs.models.discriminator import get_fc_discriminator
from libs.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from libs.utils.func import loss_calc, bce_loss
from libs.utils.loss import entropy_loss
from libs.utils.func import prob_2_entropy
from libs.utils.viz_segmask import colorize_mask

def train_advent(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.Adam(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          betas=(0.9, 0.99))

    # discriminators' optimizers
    optimizer_d_aux = optim.SGD(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer_d_main = optim.SGD(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source, input_size_source), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target, input_size_target), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for i_iter in tqdm(range(1,cfg.TRAIN.EARLY_STOP + 1)):
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)
        # d_src_prob_list = []
        # d_target_prob_list = []

        for ndx, batch_mix in enumerate(zip(trainloader, targetloader)):
            # UDA Training

            # reset optimizers
            optimizer.zero_grad()
            optimizer_d_aux.zero_grad()
            optimizer_d_main.zero_grad()

            # Train discriminator networks
            # enable training mode on discriminator networks
            for param in d_aux.parameters():
                param.requires_grad = True
            for param in d_main.parameters():
                param.requires_grad = True
            # train with source
            images_source, labels = batch_mix[0]
            pred_src_aux, pred_src_main = model(images_source.cuda(device))
            pred_src_main = interp(pred_src_main)
            if cfg.TRAIN.MULTI_LEVEL:
                pred_src_aux = interp(pred_src_aux)
                pred_src_aux = pred_src_aux.detach() # 可以避免更新到分割模型
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
                loss_d_aux_s = bce_loss(d_out_aux, source_label)
                loss_d_aux_s = loss_d_aux_s / 2 # why divide 2
                loss_d_aux_s.backward()
            else:
                loss_d_aux_s = 0
            pred_src_main = pred_src_main.detach()
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
            print(f" shape of d_out_main: {d_out_main.shape}")
            d_src_prob = F.sigmoid(d_out_main)
            # print(f"d_src_prob shape: {d_src_prob.shape}")
            # d_src_prob_list.append(d_src_prob)
            loss_d_main_s = bce_loss(d_out_main, source_label)
            loss_d_main_s = loss_d_main_s / 2 # why divide 2
            loss_d_main_s.backward()

            # train with target
            images, _ = batch_mix[1]
            pred_trg_aux, pred_trg_main = model(images.cuda(device))
            pred_src_main = interp(pred_src_main)
            if cfg.TRAIN.MULTI_LEVEL:
                pred_src_aux = interp(pred_src_aux)
                pred_trg_aux = pred_trg_aux.detach()
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                loss_d_aux_t = bce_loss(d_out_aux, target_label)
                loss_d_aux_t = loss_d_aux_t / 2 # why divide 2
                loss_d_aux_t.backward()
            else:
                loss_d_aux_t = 0
            pred_trg_main = pred_trg_main.detach()
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            d_target_prob = F.sigmoid(d_out_main)
            # print(f"d_target_prob shape: {d_target_prob.shape}")
            # d_target_prob_list.append(d_target_prob)
            loss_d_main_t = bce_loss(d_out_main, target_label)
            loss_d_main_t = loss_d_main_t / 2 # why divide 2
            loss_d_main_t.backward()
            #更新discriminator的參數
            if cfg.TRAIN.MULTI_LEVEL:
                optimizer_d_aux.step()
            optimizer_d_main.step()

            # only train segnet. Don't accumulate grads in disciminators
            for param in d_aux.parameters():
                param.requires_grad = False
            for param in d_main.parameters():
                param.requires_grad = False
            # train on source
            images_source, labels = batch_mix[0] #source domain batch
            pred_src_aux, pred_src_main = model(images_source.cuda(device))
            if cfg.TRAIN.MULTI_LEVEL:
                pred_src_aux = interp(pred_src_aux)
                loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
            else:
                loss_seg_src_aux = 0
            pred_src_main = interp(pred_src_main)
            loss_seg_src_main = loss_calc(pred_src_main, labels, device)
            loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                    + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)

            loss.backward()

            # adversarial training ot fool the discriminator
            images, _ = batch_mix[1] #target domain batch
            pred_trg_aux, pred_trg_main = model(images.cuda(device))
            if cfg.TRAIN.MULTI_LEVEL:
                pred_trg_aux = interp_target(pred_trg_aux)
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
            else:
                loss_adv_trg_aux = 0
            pred_trg_main = interp_target(pred_trg_main)
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            print(f" target domain img and source label is {source_label}, target label is {target_label}")
            print(f" d_out_main for fool target domain batch 1 prob:\n {F.sigmoid(d_out_main[0][0])}")
            loss_adv_trg_main = bce_loss(d_out_main, source_label)
            loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                    + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
            
            loss.backward()
            #更新語意分割網路
            optimizer.step()
            
            current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                            'loss_seg_src_main': loss_seg_src_main,
                            'loss_adv_trg_aux': loss_adv_trg_aux,
                            'loss_adv_trg_main': loss_adv_trg_main,
                            'loss_d_aux': loss_d_aux_s + loss_d_aux_t,
                            'loss_d_main': loss_d_main_t + loss_d_main_s,
                            "d_avg_src_prob": torch.sum(d_src_prob)/d_src_prob.numel(),
                            "d_avg_target_prob": torch.sum(d_target_prob)/d_target_prob.numel()}
            print_dict(current_losses, i_iter)
        # 每個epoch完存一次權重
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0:
            print(' taking snapshot ...')
            print(f"i_iter = {i_iter}")
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if cfg.TRAIN.MULTI_LEVEL:
                torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            print('model saved.')
            
        sys.stdout.flush() # python 的standard out 是有 buffer的，為了要強迫在腳本執行完以之前就能把buffer內的東西寫到終端機上，需要用該指令。

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def train_minent(model, trainloader, targetloader, cfg):
    ''' UDA training with minEnt
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # trainloader_iter = enumerate(trainloader)
    # targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # train on source
        for ndx, batch in enumerate(trainloader):
            images_source, labels = batch
            pred_src_aux, pred_src_main = model(images_source.cuda(device))
            if cfg.TRAIN.MULTI_LEVEL:
                pred_src_aux = interp(pred_src_aux)
                loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
            else:
                loss_seg_src_aux = 0
            pred_src_main = interp(pred_src_main)
            loss_seg_src_main = loss_calc(pred_src_main, labels, device)
            loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                    + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
            loss.backward()

        # adversarial training with minent
        for ndx, batch in enumerate(trainloader):
            images, _, = batch
            pred_trg_aux, pred_trg_main = model(images.cuda(device))
            pred_trg_aux = interp_target(pred_trg_aux)
            pred_trg_main = interp_target(pred_trg_main)
            pred_prob_trg_aux = F.softmax(pred_trg_aux)
            pred_prob_trg_main = F.softmax(pred_trg_main)

            loss_target_entp_aux = entropy_loss(pred_prob_trg_aux)
            loss_target_entp_main = entropy_loss(pred_prob_trg_main)
            loss = (cfg.TRAIN.LAMBDA_ENT_AUX * loss_target_entp_aux
                    + cfg.TRAIN.LAMBDA_ENT_MAIN * loss_target_entp_main)
            # reset optimizers
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_ent_aux': loss_target_entp_aux,
                          'loss_ent_main': loss_target_entp_main}

        print_dict(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def print_dict(current_dict, i_iter):
    list_strings = []
    for loss_name, loss_value in current_dict.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'MinEnt':
        train_minent(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
