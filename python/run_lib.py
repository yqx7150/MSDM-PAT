# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling_train as sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import os.path as osp
FLAGS = flags.FLAGS
import cv2

def save_img(img, img_path):

    img = np.clip(img*255,0,255)
    # 这行代码的作用是对图像进行处理。假设img是一个表示图像的NumPy数组，该数组中的值通常在0到1之间（表示图像像素的灰度值）。
    # 代码首先将图像的值放大了255倍（img*255），然后使用np.clip函数将数组中的值限制在0到255的范围内。这个操作可以确保图像的像素值处于合法的灰度范围（0到255之间）
    cv2.imwrite(img_path, img)
# 这行代码使用OpenCV中的imwrite函数将处理后的图像img保存到指定路径img_path中。这个函数将NumPy数组表示的图像保存为图像文件（如JPEG、PNG等）到硬盘上的指定路径。
#
# 总体而言，这段代码的作用是将图像的像素值范围调整到0到255之间，并将处理后的图像保存到指定路径
def write_kdata(Kdata,filename,name):
    # Kdata：表示处理过的数据。
    # filename：表示要保存图像文件的目录路径。
    # name：表示要保存的图像文件名
    temp = np.log(1+abs(Kdata))
    # 对输入的 Kdata 进行处理，取绝对值并取对数
    plt.axis('off')
    # 关闭图像坐标轴
    plt.imshow(abs(temp),cmap='gray')
    # 将处理后的数据以灰度图像的方式显示
    plt.savefig(osp.join(filename,name),transparent=True, dpi=128, pad_inches = 0,bbox_inches = 'tight')
#     将图像保存到指定的文件路径中。
#     其中，osp.join() 用于将文件名与路径合并，transparent=True 表示图像背景透明，dpi=128 表示图像分辨率为 128，pad_inches=0 表示不留白，
#     bbox_inches='tight' 表示调整边界框。
def write_data(data,filename,name):   
    plt.axis('off')
    plt.imshow(abs(data),cmap='gray')
    # 将处理后的数据以灰度图像的方式显示
    plt.savefig(osp.join(filename,name),transparent=True, dpi=128, pad_inches = 0,bbox_inches = 'tight')
#将图像保存到指定的文件路径中。其中，osp.join() 用于将文件名与路径合并，
# transparent=True 表示图像背景透明，dpi=128 表示图像分辨率为 128，
# pad_inches=0 表示不留白，bbox_inches='tight' 表示调整边界框
def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  # 创建实验日志的目录
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)
  # 这部分代码用于创建用于保存样本和 TensorBoard 日志的目录，
  # 并使用 TensorBoard 的 SummaryWriter 对象来记录日志
  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  # 这段代码初始化模型，并创建用于指数移动平均的 ExponentialMovingAverage 对象以及优化器对象。

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  #train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                            uniform_dequantization=config.data.uniform_dequantization)
                                              
  train_ds, eval_ds = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)

  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # 这里创建了训练和验证数据的迭代器 train_iter 和 eval_iter
  
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  # 创建用于数据标准化和反标准化的处理器。
  # Setup SDEs
  # 配置 SDEs（随机微分方程）
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  # 根据配置创建不同类型的随机微分方程对象。
  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
  # 创建了用于训练和评估模型的函数
  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  # 检查是否需要生成样本：
  #
  # 如果配置了 config.training.snapshot_sampling 为 True，
  # 则会创建用于采样的形状 (sampling_shape) 和采样函数 (sampling_fn)。
  # 这个函数 sampling.get_sampling_fn 是用于根据给定配置、随机微分方程 (sde)、
  # 数据标准化反函数 (inverse_scaler) 和采样的误差 (sampling_eps) 来获得采样函数
  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))


  # 开始训练循环：
  #
  # 在一个循环中，对每个训练步骤执行以下操作：
  #
  # a. 从数据迭代器中获取一个批次的数据。如果迭代器已耗尽，则重新初始化迭代器以获取数据。
  #
  # b. 将数据标准化处理。
  #
  # c. 执行训练步骤（使用 train_step_fn）以更新模型权重，并记录训练损失。
  #
  # d. 定期保存中间检查点，以便在需要时恢复训练。
  #
  # e. 定期报告在评估数据集上的损失情况。
  #
  # f. 定期保存检查点，并在需要时生成和保存样本
  for step in range(initial_step, num_train_steps + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    #batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    
    
    
    
    try:
      batch  = next(train_iter).float().cuda()  # 
    except StopIteration:
      train_iter = iter(train_ds)
      batch = next(train_iter).float().cuda()   # .float()
    
    
    
#    batch = torch.from_numpy(inps).to(config.device).float()
    batch = scaler(batch)
    
    #print(batch.dtype,type(batch),batch.shape)
    #assert 0
    
    #batch = batch.permute(0, 3, 1, 2)
    
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    #if step % config.training.eval_freq == 0:
      #eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      #eval_batch = eval_batch.permute(0, 3, 1, 2)
    #  eval_batch = torch.from_numpy(next(eval_iter).numpy()).to(config.device).float()
    #  eval_batch = scaler(eval_batch)
    #  eval_loss = eval_step_fn(state, eval_batch)
    #  logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
    #  writer.add_scalar("eval_loss", eval_loss.item(), step)
  
    # Save a checkpoint periodically and generate samples if needed
    # 定期保存检查点，并在需要时生成样本
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      # 生成并保存样本
      if config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        #sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        #==========================================#
        sample = sample.permute(0, 2, 3, 1).cpu().numpy() # (1, 192, 192, 322)
        #sample = sample[0,:,:,:]
        #kw_real, kw_imag = sample[0,:,:,0], sample[0,:,:,126]
        #kw_complex = kw_real+1j*kw_imag
        
        #print(np.max(np.abs(kw_complex)))
        #plt.imshow(np.log(0.1+np.abs(kw_complex)),cmap='gray')
        #plt.show()
        #assert 0
        
        
        #sample = np.log(1+np.abs(kw_complex))  #(512, 512)
        #write_kdata(sample,this_sample_dir,"sample.png")

        #==========================================#               
        #with tf.io.gfile.GFile(
        #    os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
        #  np.save(fout, sample)

        #with tf.io.gfile.GFile(
        #    os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
        
        for num_bc in range(sample.shape[3]): 
            sample_complex =  sample[0,:,:,num_bc]
            
            fout=os.path.join(this_sample_dir, "sample_" + str(num_bc) + ".png")
            save_img(sample_complex, fout )
#         这部分代码是在循环内部处理生成的样本数据。它首先将张量sample进行维度排列调整，然后通过循环遍历生成的样本，
#         并将每个通道的结果以PNG图像的形式保存到相应的文件中。
# 生成和保存样本：
#
# 如果 config.training.snapshot_sampling 为 True，
# 则在满足条件时，将生成和保存样本。
# 它在训练中周期性地使用 ema 对象来存储和恢复模型参数，
# 然后利用 sampling_fn 生成样本。
# 样本生成后，它将样本保存为图像文件并存储在 this_sample_dir 目录中，
# 每个通道的样本会分别保存在命名为 sample_<num_bc>.png 的图像文件中

def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)
  # 数据管道的构建：
  #
  # 使用 datasets.get_dataset 获取训练数据集 train_ds、评估数据集 eval_ds。
  # uniform_dequantization 参数用于指定是否使用均匀去量化
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)
  # 如果配置中启用了损失计算 (config.eval.enable_loss)，则进行以下操作：
  # 从损失模块中获取优化函数 (optimize_fn)，该函数根据配置信息管理优化过程。
  # 从配置信息中获取 continuous、likelihood_weighting 和 reduce_mean 参数。
  # 使用 losses.get_step_fn() 函数构建一个评估步骤（evaluation step）函数 (eval_step)，用于评估模型，其中包括了损失计算、梯度优化等过程。

  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      uniform_dequantization=True, evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")
  # 为了进行概率密度估计 (BPD) 的评估，创建了数据加载器：
  # 通过 datasets.get_dataset() 获取用于概率密度估计的训练集 (train_ds_bpd) 和评估集 (eval_ds_bpd)。
  # 如果配置中指定 BPD 数据集为训练集 ('train')，则 ds_bpd 设置为训练集，bpd_num_repeats 设置为 1。
  # 如果 BPD 数据集为测试集 ('test')，则 ds_bpd 设置为测试集，bpd_num_repeats 设置为 5。在计算测试集的似然时，数据集将被重复使用 5 次。
  # 否则，如果配置中指定的 BPD 数据集无法识别，则引发 ValueError。
  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)
  # 如果配置中启用了概率密度估计 (config.eval.enable_bpd)，则构建概率密度估计函数 (likelihood_fn)：
  # 使用 likelihood.get_likelihood_fn() 函数构建概率密度估计函数，该函数需要传入 SDE（随机微分方程）和反向缩放器 (inverse_scaler)。
  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  # 如果配置中启用了采样功能 (config.eval.enable_sampling)，则构建采样函数 (sampling_fn)：
  # 定义 sampling_shape 变量，表示采样形状，包括批量大小、通道数以及图像的高度和宽度。
  # 使用 sampling.get_sampling_fn() 函数构建采样函数，需要传入配置信息、SDE、采样形状、反向缩放器以及采样的 epsilon
  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)
  # 根据上一行的设置，获取 Inception 模型。
  begin_ckpt = config.eval.begin_ckpt
  # 从配置中获取评估开始的检查点编号。
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  # 在日志中记录评估开始的检查点编号。
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    # 在循环内部：
    #
    # 检查当前检查点文件是否存在，如果不存在，则等待其到达。
    # 读取检查点文件，然后等待额外的时间以确保文件已准备好读取。
    # 进行模型状态的恢复。
    # 处理评估数据集的损失函数计算，如果启用了损失计算，则遍历评估数据集并计算损失。
    # 将损失值保存到磁盘或 Google Cloud Storage 中。
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types创建一个迭代器，遍历评估数据集
      for i, batch in enumerate(eval_iter):
        # 在循环中：
        #
        # 将每个批次的图像数据转换为 PyTorch 张量，并在必要时进行数据预处理和标准化。
        # 调用 eval_step 函数计算损失，并将损失值附加到 all_losses 列表中。
        # 每计算 1000 个步骤时，在日志中记录已完成的损失评估步骤。
        # 最后，将损失值保存到磁盘或 Google Cloud 存储中，以 .npz 格式进行压缩保存，并记录损失的平均值。
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      # 如果启用了对数似然计算 config.eval.enable_bpd：
      #
      # 创建一个空列表 bpds 用于存储计算出的 bits/dim 值。
      #
      # 遍历数据集进行计算：
      #
      # 对数据集的每个重复次数（repeat）进行循环遍历。
      # 对数据集的每个批次进行遍历，并计算 bits/dim。
      # 计算得到的 bits/dim 值被扩展（extend）到 bpds 列表中。
      # 在日志中记录每个检查点、重复次数和数据批次的 bits/dim 均值。
      # 最后，将计算出的 bits/dim 值保存到磁盘或 Google Cloud 存储中，以便后续分析。
      #
      # 抱歉，由于代码中的部分函数调用和变量来自自定义库或上下文中未提供的配置，无法完全理解每行代码的作用。但是，我可以帮你理解代码的主要逻辑。
      #
      # 这部分代码段包含两个主要部分，分别是计算损失值和计算对数似然（bits/dim）值。
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

    # Generate samples and compute IS/FID/KID when enabled
    
    if config.eval.enable_sampling:
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      # 计算采样轮数，确保覆盖所有样本。
      #
      # 通过循环迭代每一轮采样：
      #
      # 在日志中记录采样的检查点和轮数。
      # 创建一个目录 this_sample_dir，用于保存采样结果。
      # 使用 sampling_fn 获取样本，并对样本数据进行处理，生成可保存的数据格式。
      # 对样本数据进行一些处理（在代码中有两种处理方式，但只选择了一种进行注释，另一种被注释掉了）。
      # 将处理后的样本数据保存到磁盘或 Google Cloud 存储中。
      for r in range(num_sampling_rounds):
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}")
        tf.io.gfile.makedirs(this_sample_dir)
        samples, n = sampling_fn(score_model)
        #=============================================================#
        samples = samples.permute(0, 2, 3, 1).cpu().numpy()
        kw_real, kw_imag = samples[0,:,:,0], samples[0,:,:,126]
        kw_complex = kw_real+1j*kw_imag
        samples = np.log(1+np.abs(kw_complex))        
        #=============================================================#
        #samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        #samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        # 在调用 Inception 网络的 TensorFlow 代码之前强制垃圾回收
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

      # Compute inception scores, FIDs and KIDs.
      # Load all statistics that have been previously computed and saved for each host
      # 强制执行垃圾回收操作，确保在调用 TensorFlow Inception 网络相关代码之前和之后进行了清理。
      #
      # 在循环之外，通过加载已经计算并保存的统计数据，收集所有的 logits（激活值）和 pool_3（Inception 网络的池化层）数据
      all_logits = []
      all_pools = []
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          if not inceptionv3:
            all_logits.append(stat["logits"])
          all_pools.append(stat["pool_3"])

      if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]
      # 如果不是 Inception V3 模型，将 all_logits 合并并截取指定数量的样本，用于后续计算 Inception Score（IS）。all_logits 存储着样本数据的 logits（激活值）。
      #
      # 将 all_pools 合并并截取指定数量的样本，用于后续计算 FID 和 KID。all_pools 存储着样本数据的池化层。
      # Load pre-computed dataset statistics.
      data_stats = evaluation.load_dataset_stats(config)
      data_pools = data_stats["pool_3"]
      # 加载预先计算的真实数据集的统计信息（data_stats）。
      #
      # 从加载的真实数据集统计信息中提取池化层数据（data_pools）
      # Compute FID/KID/IS on all samples together.
      if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      else:
        inception_score = -1
      # 计算 Inception Score（如果非 Inception V3 模型），或设置 Inception Score 为默认值 -1（如果是 Inception V3 模型）
      fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
      # Hack to get tfgan KID work for eager execution.
      tf_data_pools = tf.convert_to_tensor(data_pools)
      tf_all_pools = tf.convert_to_tensor(all_pools)
      kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
      del tf_data_pools, tf_all_pools
      # 计算 FID（Frechet Inception Distance），用于衡量生成数据与真实数据之间的差异。
      #
      # 通过 tfgan.eval.kernel_classifier_distance_from_activations 计算 KID（Kernel Inception Distance）。这里使用了一种技巧来使 KID 在 eager execution 模式下正常工作。
      #
      # 记录计算得到的 Inception Score、FID 和 KID 到日志中。
      logging.info(
        "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))

      with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                             "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
        f.write(io_buffer.getvalue())
# 将这些评估指标保存到文件中（以 .npz 格式），包括 Inception Score（IS）、FID 和 KID。
#
# 这段代码的目的是评估生成的样本数据与真实数据之间的相似程度，通过计算 Inception Score、FID 和 KID 进行量化评估，并将结果保存供进一步分析和比较。