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

"""Training and evaluation"""

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf

FLAGS = flags.FLAGS#将 flags.FLAGS 赋值给变量 FLAGS，用于管理和保存命令行参数。

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
#使用 config_flags.DEFINE_config_file() 定义一个名为 "config" 的配置文件标志，用于训练。lock_config=True 确保一旦设置了这个配置，就不能再修改。
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
# flags.DEFINE_enum() 定义一个枚举标志 "mode"，只接受列表中的值 ["train", "eval"]。该标志表示程序的运行模式，可以是训练或评估
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
# 定义一个字符串标志 "eval_folder"，表示存储评估结果的文件夹名称。默认值为 "eval"。
flags.mark_flags_as_required(["workdir", "config", "mode"])
# 将标志 "workdir"、"config" 和 "mode" 标记为必需，这意味着如果在运行脚本时没有提供这些标志，则程序将抛出错误。

def main(argv):
  if FLAGS.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # 使用 TensorFlow 的 tf.io.gfile.makedirs() 函数创建由 FLAGS.workdir 指定的工作目录
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    # 以写入模式在工作目录中打开名为 'stdout.txt' 的文件，并设置一个 StreamHandler 用于记录到该文件。
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # 获取记录器对象，将处理程序添加到记录器，并将日志级别设置为 'INFO'，以控制记录的详细程度。

    # Copy
    # Run the training pipeline
    
    print(FLAGS.config)
    run_lib.train(FLAGS.config, FLAGS.workdir)
  # 打印配置文件路径，并调用 run_lib 中的 train() 函数，将配置和工作目录路径作为参数传递。
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  # 调用 run_lib 中的 evaluate() 函数，将配置、工作目录和评估文件夹路径作为参数传递
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
# 当脚本作为主程序运行时，运行 main() 函数。使用 app.run() 方法解析命令行参数并调用 main() 函数。