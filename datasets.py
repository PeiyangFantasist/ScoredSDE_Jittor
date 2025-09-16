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
"""Return training and evaluation/test datasets from config files. For a simpler complement, only 'CIFAR-10' dataset are supported."""
import jittor as jt
import jittor.nn as nn
import jittor.transform as ts
import jittor.dataset as ds



def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x



def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution. **It only receives 4D tensor.**"""
  crop = jt.minimum(image.shape[2], image.shape[3])
  h, w = image.shape[2], image.shape[3]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  
  image = nn.resize(
    image,
    size=(resolution, resolution),
    mode='bicubic'
  )
  
  return jt.cast(image, jt.uint8)

def resize_small(image, resolution):
  """Shrink an image to the given resolution. **It only receives 4D tensor.**"""
  h, w = image.shape[2], image.shape[3]
  ratio = resolution / min(h, w)
  h = jt.round(h * ratio).cast(jt.int32).item()
  w = jt.round(w * ratio).cast(jt.int32).item()
  return nn.resize(image, (h, w))


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """
    创建 CIFAR-10 训练和评估数据集加载器（PyTorch 版本）
    
    Args:
        config: 配置对象，需包含数据相关参数（image_size, centered, random_flip 等）
        uniform_dequantization: 是否启用均匀反量化（将整数像素值转换为连续分布）
        evaluation: 是否为评估模式（控制数据增强和打乱）
    
    Returns:
        train_loader: 训练集数据加载器
        eval_loader: 评估集数据加载器
    """
    # 批次大小（评估时使用评估批次大小）
    batch_size = config.eval.batch_size if evaluation else config.training.batch_size
    
    # 数据预处理管道
    transform_list = []
    
    # 1. 调整图像大小（CIFAR-10 原始尺寸为 32x32，若配置不同则缩放）
    if config.data.image_size != 32:
        transform_list.append(nn.Resize((config.data.image_size, config.data.image_size)))  # 抗锯齿缩放
    
    # 2. 训练时随机水平翻转（评估时不使用）
    if config.data.random_flip and not evaluation:
        transform_list.append(ts.RandomHorizontalFlip())
    
    # 3. 转换为张量（将 PIL 图像转为 [0, 1] 范围的 tensor）
    transform_list.append(ts.ToTensor())
    
    # 4. 均匀反量化（模拟连续分布，仅用于整数像素值）
    if uniform_dequantization:
        # 对 [0,1] 范围的 tensor 反向缩放至 [0,255]，加随机噪声后再缩放到 [0,1)
        transform_list.append(lambda x: (x * 255.0 + jt.rand_like(x)) / 256.0)
    
    # 5. 归一化（根据 centered 参数决定范围）
    if config.data.centered:
        # 归一化到 [-1, 1]（配合 ToTensor 后的 [0,1] 范围）
        transform_list.append(ts.Normalize(mean=[0.5, 0.5, 0.5], 
                                                  std=[0.5, 0.5, 0.5]))
    else:
        # 保留 [0,1] 范围（无需额外归一化，ToTensor 已处理）
        pass
    
    # 组合所有变换
    transform = ts.Compose(transform_list)
    
    # 加载 CIFAR-10 数据集
    train_dataset = ds.CIFAR10(
        root=config.data.data_dir,  # 数据存储路径（需在 config 中定义）
        train=True,
        download=True,  # 首次运行时自动下载
        transform=transform
    )
    
    eval_dataset = ds.CIFAR10(
        root=config.data.data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = ds.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not evaluation,  # 训练时打乱，评估时不打乱
        num_workers=config.data.num_workers,  # 多线程加载（需在 config 中定义）
        pin_memory=True,  # 加速 GPU 传输
        drop_last=True  # 丢弃最后一个不完整批次
    )
    
    eval_loader = ds.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False  # 评估时保留所有样本
    )
    
    return train_loader, eval_loader