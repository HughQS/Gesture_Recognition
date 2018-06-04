# -*- coding: utf-8 -*-
"""
Created on 2018 3.26
@author: hugh
"""

import tensorflow as tf
slim = tf.contrib.slim

from nets.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
from nets.resnet_v1 import resnet_arg_scope, resnet_v1_50
# resnet_v2_50, resnet_v2_101, resnet_v2_152
from nets.resnet_v2 import resnet_arg_scope, resnet_v2_50


class NetGraph(object):
	
	def get_net(self, net_model, X, num_classes, k_prob, is_training):
		"""根据参数返回指定网络计算图
		Args:
			net_model: 网络计算图名称
			X：输入
			num_classes：类别数量
			k_prob：dropout的保留概率
			is_training： 是否是训练
		"""
		if net_model == "inception_resnet_v2":
			self.net = self._inception_resnet_v2(X, num_classes, k_prob, is_training)
		elif net_model == "resnet_v2_50":
			self.net = self._resnet_v2_50(X, num_classes, k_prob, is_training)
		elif net_model == "resnet_v1_50":
			self.net = self._resnet_v1_50(X, num_classes, k_prob, is_training)
		return self.net

	def _inception_resnet_v2(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
		arg_scope = inception_resnet_v2_arg_scope()
		with slim.arg_scope(arg_scope):
			net, end_points = inception_resnet_v2(X, num_classes=num_classes, dropout_keep_prob=dropout_keep_prob,
                        is_training=is_train)
		return net
		
	def _resnet_v1_50(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
		arg_scope = resnet_arg_scope()
		with slim.arg_scope(arg_scope):
			net, end_points = resnet_v1_50(X, is_training=is_train)
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
			with tf.variable_scope('Logits_out'):
				net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out0')
				net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out0')
				net = slim.conv2d(net, 200, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out1')
				net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out1')
				net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out2')
				net = tf.squeeze(net,[1,2], name='SpatialSqueeze')
		return net

	def _resnet_v2_50(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
		arg_scope = resnet_arg_scope()
		with slim.arg_scope(arg_scope):
			net, end_points = resnet_v2_50(X, is_training=is_train)
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
			with tf.variable_scope('Logits_out'):
				net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out0')
				net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out0')
				net = slim.conv2d(net, 200, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out1')
				net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out1')
				net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out2')
				net = tf.squeeze(net,[1,2], name='SpatialSqueeze')
		return net