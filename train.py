# -*- coding: utf-8 -*-
"""
Created on 2018 3.26
@author: hugh
"""

import tensorflow as tf
import numpy as np
import os
import config
from keras.utils import np_utils
from load_image import get_next_batch_from_path, shuffle_train_data, LoadImage
import build_net

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

slim = tf.contrib.slim

def train(train_data, train_label, valid_data, valid_label, train_n, valid_n, IMAGE_HEIGHT,
		  IMAGE_WIDTH, learning_rate, num_classes, epoch, EARLY_STOP_PATIENCE, DISPLAY_TRAIN_EVERY,
		  batch_size=64, keep_prob=0.8, net_model="resnet_v2_50", num_checkpoints=3,
		  checkpoint_path="model/resnet_v2_50"):
	"""训练函数
	Args:
		train_data: 训练数据
		train_label：训练标签
		valid_data：验证数据
		valid_label：验证标签
		train_n：训练数据的数据量
		valid_n：验证数据的数据量
		IMAGE_HEIGHT：训练图片的高度
		IMAGE_WIDTH:训练图片的宽度
		learning_rate:学习率
		num_classes:类别数量
		epoch:训练轮数
		EARLY_STOP_PATIENCE：停止训练的容忍度		
		DISPLAY_TRAIN_EVERY:显示训练的批次
		batch_size: 批量的大小
		keep_prob: dropout的保留概率
		net_model: 网络计算图的名称
		num_checkpoints: 保留训练的模型的个数
		checkpoint_path：训练模型的保存路径	
	"""
	# 定义输入、输出等占位符
	X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
	Y = tf.placeholder(tf.float32, [None, num_classes])
	is_training = tf.placeholder(tf.bool, name='is_training')
	k_prob = tf.placeholder(tf.float32) # dropout

	# 定义网络计算图
	Net = build_net.NetGraph()
	net = Net.get_net(net_model, X, num_classes, k_prob, is_training)

	# 定义损失函数
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = net))
	
	# 定义优化方式
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
		
	predict = tf.reshape(net, [-1, num_classes])
	max_idx_p = tf.argmax(predict, 1)
	max_idx_l = tf.argmax(Y, 1)
	correct_pred = tf.equal(max_idx_p, max_idx_l)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# tensorboard
	with tf.name_scope('tmp/'):
		tf.summary.scalar('loss', loss)
		tf.summary.scalar('accuracy', accuracy)
	summary_op = tf.summary.merge_all()
	#------------------------------------------------------------------------------------#
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	# 定义训练和验证log保存
	def make_dir(log_dir):
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
	train_log_dir = net_model + '_train_log'
	valid_log_dir = net_model + '_valid_log'
	make_dir(train_log_dir)
	make_dir(valid_log_dir)
	
	train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
	valid_writer = tf.summary.FileWriter(valid_log_dir, sess.graph)
	
	saver_net = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
	if os.path.exists(checkpoint_path):
		restore_flag = False
		for (path, dirnames, filenames) in os.walk(checkpoint_path):
			if len(filenames) != 0:
				restore_flag = True
		if restore_flag:
			saver_net.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
	else:
		os.makedirs(checkpoint_path)

	# early stopping
	best_loss_valid = np.inf
	best_acc_valid = 0
	best_valid_epoch = 0

	for epoch_i in range(epoch):
		print('Epoch===================================>: {:>2}'.format(epoch_i))
		for batch_i in range(int(train_n/batch_size)):
			images_train, labels_train = get_next_batch_from_path(train_data, train_label, batch_i, IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=True)
			los, _ = sess.run([loss,optimizer], feed_dict={X: images_train, Y: labels_train, k_prob:keep_prob, is_training:True})
			if batch_i % DISPLAY_TRAIN_EVERY == 0:
				loss_, acc_, summary_str = sess.run([loss, accuracy, summary_op], feed_dict={X: images_train, Y: labels_train, k_prob:1.0, is_training:False})
				train_writer.add_summary(summary_str, global_step=((int(train_n/batch_size))*epoch_i+batch_i))
				print('Epoch: {:>2}: Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(epoch_i, batch_i, loss_, acc_))

		valid_ls = 0
		valid_acc = 0
		for batch_i in range(int(valid_n/batch_size)):
			images_valid, labels_valid = get_next_batch_from_path(valid_data, valid_label, batch_i, IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=False)
			epoch_ls, epoch_acc = sess.run([loss, accuracy], feed_dict={X: images_valid, Y: labels_valid, k_prob:1.0, is_training:False})
			valid_ls = valid_ls + epoch_ls
			valid_acc = valid_acc + epoch_acc
		loss_valid = valid_ls / int(valid_n / batch_size)
		acc_valid = valid_acc / int(valid_n / batch_size)
		print('Epoch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(epoch_i, loss_valid, acc_valid))
		if acc_valid > 0.9:
			if acc_valid > best_acc_valid:
				saver_net.save(sess, checkpoint_path, global_step=epoch_i, write_meta_graph=False)
				best_acc_valid = acc_valid
			elif acc_valid == best_acc_valid and loss_valid < best_loss_valid:
				saver_net.save(sess, checkpoint_path, global_step=epoch_i, write_meta_graph=False)
				best_loss_valid = loss_valid
				best_valid_epoch = epoch_i
		if best_valid_epoch + EARLY_STOP_PATIENCE < epoch_i:
			print("Early stopping.")
			print("Best valid loss was {:.6f} at epoch {}.".format(best_loss_valid, best_valid_epoch))
			break
		print('>>>>>>>>>>>>>>>>>>>shuffle train_data<<<<<<<<<<<<<<<<<')
		# 每个epoch，重新打乱一次训练集：
		train_data, train_label = shuffle_train_data(train_data, train_label)
	sess.close()

if __name__ == '__main__':

	IMAGE_HEIGHT = 64
	IMAGE_WIDTH = 64

	print ("-----------------------------load_image.py start--------------------------")
	# 准备训练数据
	all_image = LoadImage("./")
	train_data, train_label, valid_data, valid_label= all_image.gen_train_valid_image()
	# 训练数据的总有效数量
	image_n = all_image.image_n	
	print ("训练数据的总有效数量:{}".format(image_n))
	# 获取训练数据的类别数
	num_classes = all_image.num_classes
	print ("训练数据的的总类别数:{}".format(num_classes))
	print ("配置的总类别数:{}".format(config.num_classes))
	
	train_n = all_image.train_n
	valid_n = all_image.valid_n
	# ont-hot
	train_label = np_utils.to_categorical(train_label, num_classes)
	valid_label = np_utils.to_categorical(valid_label, num_classes)

	print ("-----------------------------train.py start--------------------------")
	train(train_data, train_label, valid_data, valid_label, train_n, valid_n,
		IMAGE_HEIGHT, IMAGE_WIDTH, learning_rate=config.learning_rate, num_classes=config.num_classes, epoch=config.epoch,
		EARLY_STOP_PATIENCE=config.EARLY_STOP_PATIENCE, DISPLAY_TRAIN_EVERY=config.DISPLAY_TRAIN_EVERY,batch_size=config.batch_size,
		keep_prob=config.keep_prob,net_model=config.net_model, num_checkpoints=config.num_checkpoints,
		checkpoint_path=config.checkpoint_path)
