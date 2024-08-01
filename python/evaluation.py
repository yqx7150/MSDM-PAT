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

"""Utility functions for computing FID/Inception scores."""

import jax
import numpy as np
import six
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(inceptionv3=False):
  if inceptionv3:
    return tfhub.load(
      'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
  else:
    return tfhub.load(INCEPTION_TFHUB)
# get_inception_model(inceptionv3=False):
# 这是一个函数，它有一个布尔类型的参数 inceptionv3，默认为 False。
#
# inceptionv3 参数用于决定加载哪个版本的 Inception 模型。
# 如果 inceptionv3 参数为 True，则返回预训练的 Inception V3 模型，使用了 TensorFlow Hub 加载。
# 如果 inceptionv3 参数为 False，则返回预先定义的常量 INCEPTION_TFHUB（假设已在代码中定义），并使用 TensorFlow Hub 加载该模型。
# 返回值
# 如果 inceptionv3 参数为 True，则返回加载的 Inception V3 模型。
# 如果 inceptionv3 参数为 False，则返回使用指定 URL（在 INCEPTION_TFHUB 中定义）加载的 Inception 模型。
# 这个函数的目的是根据参数加载并返回不同版本的 Inception 模型。

def load_dataset_stats(config):
  """Load the pre-computed dataset statistics."""
  if config.data.dataset == 'CIFAR10':
    filename = 'assets/stats/cifar10_stats.npz'
  elif config.data.dataset == 'CELEBA':
    filename = 'assets/stats/celeba_stats.npz'
  elif config.data.dataset == 'LSUN':
    filename = f'assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    stats = np.load(fin)
    return stats


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
  # output_fields：字符串、列表或 None。如果存在，则假设模块输出一个字典，并选择该字段。
  # inception_model：从 TFHub 加载的模型。
  # return_tensor：如果为 True，则返回单个张量而不是字典
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Args:
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    inception_model: A model loaded from TFHub.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]
  # 如果 output_fields 是字符串类型，则将其转换为列表类型，否则保持不变
  def _classifier_fn(images):
    # 定义了名为 _classifier_fn 的内部函数，它接收图像张量作为输入。
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    #     如果提供了 output_fields，则从模型输出中选择特定字段。
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)
  # 返回一个通过 tf.compat.v1.layers.flatten 函数处理过的输出结构。这个函数返回了一个接受图像张量并返回模型输出的函数
  return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
  """Running the inception network. Assuming input is within [0, 255]."""
  # inputs：输入数据
  # inception_model：Inception 模型
  # num_batches：批次数量，默认为 1
  # inceptionv3：一个布尔值，指示是否为 Inception v3 模型
  if not inceptionv3:
    inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
  # 如果不是 Inception v3 模型，则将输入数据转换为浮点型，并进行归一化
  else:
    inputs = tf.cast(inputs, tf.float32) / 255.
  # 将输入数据转换为浮点型并进行归一化（范围在 [0, 1] 之间）。
  return tfgan.eval.run_classifier_fn(
    inputs,
    num_batches=num_batches,
    classifier_fn=classifier_fn_from_tfhub(None, inception_model),
    dtypes=_DEFAULT_DTYPES)
# 调用 tfgan.eval.run_classifier_fn 函数并返回结果。此函数用于评估分类器函数，计算给定输入的模型输出。函数的参数如下：
#
# inputs：输入数据
# num_batches：批次数量
# classifier_fn：通过 classifier_fn_from_tfhub 生成的分类器函数
# dtypes：默认数据类型
# 总体而言，该函数是一个用于运行 Inception 网络并返回输出结果的功能

@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
  # 定义了一个名为 run_inception_distributed 的函数，用于将 Inception 网络的计算分发到所有可用的 TPU 上
  """Distribute the inception network computation to all available TPUs.

  Args:
    input_tensor: The input images. Assumed to be within [0, 255].
    inception_model: The inception network model obtained from `tfhub`.
    num_batches: The number of batches used for dividing the input.
    inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

  Returns:
    A dictionary with key `pool_3` and `logits`, representing the pool_3 and
      logits of the inception network respectively.
  """
  num_tpus = jax.local_device_count()
  # 获取当前设备上的 TPU 数量
  input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  pool3 = []
  logits = [] if not inceptionv3 else None
  device_format = '/TPU:{}' if 'TPU' in str(jax.devices()[0]) else '/GPU:{}'
  # 根据设备类型创建设备格式字符串，如果第一个设备是 TPU，则格式为 '/TPU:{}'，否则为 '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      # 将操作指定在相应的设备上进行计算
      tensor_on_device = tf.identity(tensor)
      # 将输入张量复制到指定设备上。
      res = run_inception_jit(
        tensor_on_device, inception_model, num_batches=num_batches,
        inceptionv3=inceptionv3)
      # 使用 JIT 编译的函数 run_inception_jit 对每个设备上的输入数据执行 Inception 网络，并获取输出结果。
      if not inceptionv3:
        pool3.append(res['pool_3'])
        logits.append(res['logits'])  # pytype: disable=attribute-error
      else:
        pool3.append(res)

  with tf.device('/CPU'):
    return {
      'pool_3': tf.concat(pool3, axis=0),
      'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
    }
