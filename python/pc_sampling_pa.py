from dataclasses import dataclass, field

import io
import csv
import numpy as np
# import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation7
from utils import restore_checkpoint

sns.set(font_scale=2)
sns.set(style="whitegrid")

import cv2

from matplotlib.image import imread

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      EulerMaruyamaPredictor,
                      AncestralSamplingPredictor,
                      NoneCorrector,
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse

# from skimage.measure import compare_psnr,compare_ssim


import math
import scipy.io as io

import torch

torch.cuda.empty_cache()

# @title Load the score-based model
sde = 'VESDE'  # @param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
    from configs.ve import SIAT_kdata_ncsnpp_test as configs_1  # 修改config
    from configs.ve import SIAT_kdata_ncsnpp_test2 as configs_2  # 修改config


    # TODO:要使用的模型(的路径)
    ckpt_filename_1 = "exp1/checkpoints/checkpoint_20.pth"  ###model1，360
    ckpt_filename_2 = "exp2/checkpoints/checkpoint_20.pth"  ###model2，240
    print(ckpt_filename_1)
    print(ckpt_filename_2)
    if not os.path.exists(ckpt_filename_1):
        print('!!!!!!!!!!!!!!' + ckpt_filename_1 + ' not exists')
        assert False
    if not os.path.exists(ckpt_filename_2):
        print('!!!!!!!!!!!!!!' + ckpt_filename_2 + ' not exists')
        assert False
    config_1 = configs_1.get_config()
    config_2 = configs_2.get_config()
    sde_1 = VESDE(sigma_min=config_1.model.sigma_min, sigma_max=config_1.model.sigma_max, N=config_1.model.num_scales)
    sde_2 = VESDE(sigma_min=config_2.model.sigma_min, sigma_max=config_2.model.sigma_max, N=config_2.model.num_scales)
    sampling_eps = 1e-5

#model1
batch_size = 1 #@param {"type":"integer"}
config_1.training.batch_size = batch_size
config_1.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config_1)
scaler = datasets.get_data_scaler(config_1)
inverse_scaler = datasets.get_data_inverse_scaler(config_1)
score_model_1 = mutils.create_model(config_1)

optimizer = get_optimizer(config_1, score_model_1.parameters())
ema = ExponentialMovingAverage(score_model_1.parameters(),
                               decay=config_1.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model_1, ema=ema)

state = restore_checkpoint(ckpt_filename_1, state, config_1.device)
ema.copy_to(score_model_1.parameters())

#model2
batch_size = 1 #@param {"type":"integer"}
config_2.training.batch_size = batch_size
config_2.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config_2)
scaler = datasets.get_data_scaler(config_2)
inverse_scaler = datasets.get_data_inverse_scaler(config_2)
score_model_2 = mutils.create_model(config_2)

optimizer = get_optimizer(config_2, score_model_2.parameters())
ema = ExponentialMovingAverage(score_model_2.parameters(),
                               decay=config_2.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model_2, ema=ema)

state = restore_checkpoint(ckpt_filename_2, state, config_2.device)
ema.copy_to(score_model_2.parameters())
# @title PC inpainting

predictor = ReverseDiffusionPredictor  # @param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector  # @param ["LangevinCorrector", "Anneb  ledLangevinDynamics", "None"] {"type": "raw"}
snr = 0.21  # 0.07#0.075 #0.16 #@param {"type": "number"}
n_steps = 1  # @param {"type": "integer"}
probability_flow = False  # @param {"type": "boolean"}

pc_inpainter = controllable_generation7.get_pc_inpainter(sde_1,sde_2,
                                                         predictor, corrector,
                                                         inverse_scaler,
                                                         snr=snr,
                                                         n_steps=n_steps,
                                                         probability_flow=probability_flow,
                                                         continuous=config_1.training.continuous,
                                                         denoise=True)



def full_pc(dimg):
    k_w = dimg
    #ori_data = datap
    X, data = pc_inpainter(score_model_1, score_model_2 ,k_w)
    return X, data


def save_img(img, img_path):
    img = np.clip(img * 255, 0, 255)  ##最小值0, 最大值255

    cv2.imwrite(img_path, img)


def write_Data(filedir, model_num, dic1):
    # filedir="result.txt"
    with open(os.path.join(filedir), "a+") as f:  # a+
        # f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.writelines(str(model_num) + ' ' + str(dic1))
        f.write('\n')


psnr_all = []
ssim_all = []
mse_all = []
nmse_all = []
import scipy.io as scio


# TODO：输入的一张好图

path ='./lzdata/test'



for aaa in sorted(os.listdir(path)):
    
    file_path = os.path.join(path, aaa)  # 输入指定的全采图
    good_input = scio.loadmat(file_path)['sensor_data']
    #good_input = cv2.imread(file_path, 0)  # 文件的图片
    data2 = np.array(good_input)
    #data = good_input / 255.
    #print(data.shape)
    ####修改代码666
    # datapad = np.zeros((512,512),dtype=np.float64)
    datapad = np.zeros((256, 256), dtype=np.float64)
    #datapad = data  ####这里就是测试的数据，展成了256 成了测试的那张图片归一化之后的图
    datapad = good_input
    #dimg = sde.prior_sampling((1, 256, 256))  # 注意这里。尺寸
    # dimg = cv2.imread('./results/inpainter2/GT_403.jpg_256.png',0)
    # dimg = np.zeros((1,256,256))
    #dimg = dimg.squeeze(0)  # 原来有.permute(1,2,0)
    #dimg = dimg.numpy()  # 处理一下
    #print(dimg.shape)
    ####save_img(dimg, './results/inpainter2/'+aaa)#dimg是一个噪声
    #psnr_zero = compare_psnr(255. * dimg, 255. * datapad, data_range=255)
    #ssim_zero = compare_ssim(dimg, datapad, data_range=1, multichannel=True)
    # print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)

    recon, data = full_pc(datapad)  # 预测器校正器操作
    # print(psnr_all)
    # print('00000000000000*******&&&&&&')
    # print(ssim_all)
    # print(recon.shape,recon.max(),recon.min())
    recon = recon.cpu()
    recon = recon.numpy()
    recon = (recon - recon.min()) / (recon.max() - recon.min())
    # recon = np.clip(recon,0,1)
    recon1 = 255. * recon
    print('#######397')
    print(recon1.shape)
    print(data.shape)
    recon = recon.squeeze(0)  # 看看网络出来的是什么
    # recon = recon.transpose(2,1,0)
    #####顺序该一下。
    recon = recon.transpose(1, 2, 0)

    # save_img(recon,'./recon1.png')
    # save_img(data,'./data1.png')
    # save_img(recon,os.path.join('./results/Rec/', aaa))
    # io.savemat(os.path.join('./results/Rec/',aaa +'.mat'),{'data':recon})

    ###xiugaiyidiandian###
    # write_Data("./results/Rec/result_last.txt", 'model'+ckpt_filename + '_' + aaa, {"psnr":psnr_all, "ssim":ssim_all})
    # 666
    # good_img = data[128:384,128:384]
    # bad_img = recon[128:384,128:384]
    # with open(os.path.join('./结果/新结果',aaa),"a+") as f:#a+
    # f.writelines(' '+str(round(psnr, 4))+' '+str(round(ssim, 4)))
    # f.write('\n')
    good_img = data
    bad_img = recon
    cv2.imwrite(os.path.join('./results/good/', aaa), 255. * good_img)
    cv2.imwrite(os.path.join('./results/bad/', aaa), 255. * bad_img)
    # cv2.imwrite('./wwresult/good.png',255.*good_img)
    # cv2.imwrite('./wwresult/bad.png',255.*bad_img)
    recon = bad_img
    data = good_img
