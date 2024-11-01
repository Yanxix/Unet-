# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import UNet_3Plus
from metrics import dice_coef, batch_iou, mean_iou, iou_score, ppv, sensitivity
import losses
from utils import str2bool, count_params
import joblib  # 这里修正了 sklearn.externals.joblib 为直接 import joblib
import SimpleITK as sitk

wt_dices = []
tc_dices = []
et_dices = []
wt_sensitivities = []
tc_sensitivities = []
et_sensitivities = []
wt_ppvs = []
tc_ppvs = []
et_ppvs = []
wt_Hausdorf = []
tc_Hausdorf = []
et_Hausdorf = []

# 新增IoU指标
wt_ious = []
tc_ious = []
et_ious = []


# 修改后的 hausdorff_distance 函数，跳过全零图像的 Hausdorff 距离计算
def hausdorff_distance(lT, lP):
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)

    # 检查是否有前景（非零像素）
    if np.count_nonzero(lT) == 0 or np.count_nonzero(lP) == 0:
        # 如果任意一个为全零，返回一个默认的Hausdorff距离，比如无穷大
        return float('inf')

    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    return hausdorffcomputer.GetAverageHausdorffDistance()


def CalculateWTTCET(wtpbregion, wtmaskregion, tcpbregion, tcmaskregion, etpbregion, etmaskregion):
    # 开始计算WT
    dice = dice_coef(wtpbregion, wtmaskregion)
    wt_dices.append(dice)
    ppv_n = ppv(wtpbregion, wtmaskregion)
    wt_ppvs.append(ppv_n)
    Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
    wt_Hausdorf.append(Hausdorff)
    sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
    wt_sensitivities.append(sensitivity_n)
    iou_n = iou_score(wtpbregion, wtmaskregion)  # 添加WT的IoU计算
    wt_ious.append(iou_n)

    # 开始计算TC
    dice = dice_coef(tcpbregion, tcmaskregion)
    tc_dices.append(dice)
    ppv_n = ppv(tcpbregion, tcmaskregion)
    tc_ppvs.append(ppv_n)
    Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
    tc_Hausdorf.append(Hausdorff)
    sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
    tc_sensitivities.append(sensitivity_n)
    iou_n = iou_score(tcpbregion, tcmaskregion)  # 添加TC的IoU计算
    tc_ious.append(iou_n)

    # 开始计算ET
    dice = dice_coef(etpbregion, etmaskregion)
    et_dices.append(dice)
    ppv_n = ppv(etpbregion, etmaskregion)
    et_ppvs.append(ppv_n)
    Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
    et_Hausdorf.append(Hausdorff)
    sensitivity_n = sensitivity(etpbregion, etmaskregion)
    et_sensitivities.append(sensitivity_n)
    iou_n = iou_score(etpbregion, etmaskregion)  # 添加ET的IoU计算
    et_ious.append(iou_n)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="batchsize",
                        help='model name')
    parser.add_argument('--mode', default=None,
                        help='')

    args = parser.parse_args()

    return args


def GetPatchPosition(PatchPath):
    npName = os.path.basename(PatchPath)
    firstName = npName
    overNum = npName.find(".npy")
    npName = npName[0:overNum]
    PeopleName = npName
    overNum = npName.find("_")
    while (overNum != -1):
        npName = npName[overNum + 1:len(npName)]
        overNum = npName.find("_")
    overNum = firstName.find("_" + npName + ".npy")
    PeopleName = PeopleName[0:overNum]
    return int(npName), PeopleName


# 修改加载模型权重以处理 "module." 前缀问题
def load_model_weights(model, model_path, multi_gpu=False):
    state_dict = torch.load(model_path)

    # 如果模型是通过 DataParallel 训练的，并且想在多 GPU 环境下继续使用
    if multi_gpu:
        model = torch.nn.DataParallel(model)  # 包装模型为DataParallel
        model.load_state_dict(state_dict)
    else:
        # 处理 DataParallel 模型的 'module.' 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # 去掉 'module.' 前缀
                new_key = k[7:]
            else:
                new_key = k
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)

    return model  # 返回模型


def main():
    val_args = parse_args()

    args = joblib.load('models/batchsize/args.pkl')
    if not os.path.exists('output/batchsize' ):
        os.makedirs('output/batchsize' )
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')
    joblib.dump(args, 'models/batchsize/args.pkl')

    # 创建模型
    print("=> creating model %s" % args.arch)
    model = UNet_3Plus.__dict__[args.arch](args)

    model = model.cuda()

    # 检查是否需要多 GPU 推理
    multi_gpu = torch.cuda.device_count() > 1

    # 加载模型权重
    model = load_model_weights(model, 'models/batchsize/model.pth' , multi_gpu=multi_gpu)

    # 数据加载代码（保持不变）
    img_paths = glob(r'D:\deeplearning\BrainTumorSegmentation-main\data\testImage\*')
    mask_paths = glob(r'D:\deeplearning\BrainTumorSegmentation-main\data\testMask\*')

    val_img_paths = img_paths
    val_mask_paths = mask_paths

    model.eval()

    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    savedir = 'output/batchsize/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            startFlag = 1
            for mynum, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                input = input.cuda()
                output = model(input)
                output = torch.sigmoid(output).data.cpu().numpy()
                target = target.data.cpu().numpy()
                img_paths = val_img_paths[args.batch_size * mynum:args.batch_size * (mynum + 1)]
                for i in range(output.shape[0]):
                    if (startFlag == 1):  # 第一个块的处理
                        startFlag = 0
                        PatchPosition, NameNow = GetPatchPosition(img_paths[i])
                        LastName = NameNow
                        OnePeople = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneMask = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneWT = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneTC = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneET = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneWTMask = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneTCMask = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneETMask = np.zeros([155, 160, 160], dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i, 0, idx, idy] > 0.5:  # WT拼接
                                    OneWT[PatchPosition, idx, idy] = 1
                                if output[i, 1, idx, idy] > 0.5:  # TC拼接
                                    OneTC[PatchPosition, idx, idy] = 1
                                if output[i, 2, idx, idy] > 0.5:  # ET拼接
                                    OneET[PatchPosition, idx, idy] = 1
                        OneWTMask[PatchPosition, :, :] = target[i, 0, :, :]
                        OneTCMask[PatchPosition, :, :] = target[i, 1, :, :]
                        OneETMask[PatchPosition, :, :] = target[i, 2, :, :]
                    PatchPosition, NameNow = GetPatchPosition(img_paths[i])
                    if (NameNow != LastName):
                        CalculateWTTCET(OneWT, OneWTMask, OneTC, OneTCMask, OneET, OneETMask)
                        for idz in range(OneWT.shape[0]):
                            for idx in range(OneWT.shape[1]):
                                for idy in range(OneWT.shape[2]):
                                    if (OneWT[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 2
                                    if (OneTC[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 1
                                    if (OneET[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 4
                        SavePeoPle = np.zeros([155, 240, 240], dtype=np.uint8)
                        SavePeoPle[:, 40:200, 40:200] = OnePeople[:, :, :]
                        saveout = sitk.GetImageFromArray(SavePeoPle)
                        sitk.WriteImage(saveout, savedir + LastName + ".nii.gz")

                        LastName = NameNow
                        OnePeople = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneMask = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneWT = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneTC = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneET = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneWTMask = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneTCMask = np.zeros([155, 160, 160], dtype=np.uint8)
                        OneETMask = np.zeros([155, 160, 160], dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i, 0, idx, idy] > 0.5:  # WT拼接
                                    OneWT[PatchPosition, idx, idy] = 1
                                if output[i, 1, idx, idy] > 0.5:  # TC拼接
                                    OneTC[PatchPosition, idx, idy] = 1
                                if output[i, 2, idx, idy] > 0.5:  # ET拼接
                                    OneET[PatchPosition, idx, idy] = 1
                        OneWTMask[PatchPosition, :, :] = target[i, 0, :, :]
                        OneTCMask[PatchPosition, :, :] = target[i, 1, :, :]
                        OneETMask[PatchPosition, :, :] = target[i, 2, :, :]
                    if (NameNow == LastName):
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i, 0, idx, idy] > 0.5:  # WT拼接
                                    OneWT[PatchPosition, idx, idy] = 1
                                if output[i, 1, idx, idy] > 0.5:  # TC拼接
                                    OneTC[PatchPosition, idx, idy] = 1
                                if output[i, 2, idx, idy] > 0.5:  # ET拼接
                                    OneET[PatchPosition, idx, idy] = 1
                        OneWTMask[PatchPosition, :, :] = target[i, 0, :, :]
                        OneTCMask[PatchPosition, :, :] = target[i, 1, :, :]
                        OneETMask[PatchPosition, :, :] = target[i, 2, :, :]
                    if mynum == len(val_loader) - 1:
                        CalculateWTTCET(OneWT, OneWTMask, OneTC, OneTCMask, OneET, OneETMask)
                        for idz in range(OneWT.shape[0]):
                            for idx in range(OneWT.shape[1]):
                                for idy in range(OneWT.shape[2]):
                                    if (OneWT[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 2
                                    if (OneTC[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 1
                                    if (OneET[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 4
                        SavePeoPle = np.zeros([155, 240, 240], dtype=np.uint8)
                        SavePeoPle[:, 40:200, 40:200] = OnePeople[:, :, :]
                        saveout = sitk.GetImageFromArray(SavePeoPle)
                        sitk.WriteImage(saveout, savedir + LastName + ".nii.gz")

            torch.cuda.empty_cache()

    print('WT Dice: %.4f' % np.mean(wt_dices))
    print('TC Dice: %.4f' % np.mean(tc_dices))
    print('ET Dice: %.4f' % np.mean(et_dices))
    print("=============")
    print('WT PPV: %.4f' % np.mean(wt_ppvs))
    print('TC PPV: %.4f' % np.mean(tc_ppvs))
    print('ET PPV: %.4f' % np.mean(et_ppvs))
    print("=============")
    print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
    print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
    print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
    print("=============")
    print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
    print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
    print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
    print("=============")
    print('WT IoU: %.4f' % np.mean(wt_ious))
    print('TC IoU: %.4f' % np.mean(tc_ious))
    print('ET IoU: %.4f' % np.mean(et_ious))
    print("=============")


if __name__ == '__main__':
    main()
