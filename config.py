# -*- coding: utf-8 -*-
"""
Created on 2018 3.26
@author: hugh
"""

# inception_resnet_v2:299 
# resnet_v2:224
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
num_classes = 6
EARLY_STOP_PATIENCE = 100
DISPLAY_TRAIN_EVERY = 20
# epoch
epoch = 1000
batch_size = 64
# 模型的学习率
learning_rate = 0.001
keep_prob = 0.8
# 设置训练样本的占总样本的比例：
train_rate = 0.9


# 选择需要的模型
net_model="resnet_v2_50"

# 迁移学习模型参数，
checkpoint_path="model/resnet_v2_50/"
num_checkpoints = 1
