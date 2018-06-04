# -*- coding: utf-8 -*-
"""
Created on 2018 3.26
@author: hugh
"""
import tensorflow as tf
import os
import math
import cv2
from load_image import LoadImage,get_next_batch_from_test_path
import build_net
import config
from sklearn.metrics import accuracy_score,precision_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def predict(test_imgs, test_n, batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, num_classes, net_model="resnet_v2_50",
				num_checkpoints=3,checkpoint_path="model/resnet_v2_50"):
	"""预测函数
	Args:
		test_imgs: 测试图片
		IMAGE_HEIGHT: 测试图片高度
		IMAGE_WIDTH: 测试图片宽度
		num_classes: 图片的类别数量
		net_model: 网络名称
		num_checkpoints: 网络模型保存的数量
		checkpoint_path: 网络模型保存的路径
	Returns：
		index：识别出的图片对应的标签
	"""
	X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
	is_training = tf.placeholder(tf.bool, name='is_training')
	k_prob = tf.placeholder(tf.float32) # dropout

	# 定义模型
	Net = build_net.NetGraph()
	net = Net.get_net(net_model, X, num_classes, k_prob, is_training)

	predict = tf.reshape(net, [-1, num_classes])
	max_idx_p = tf.argmax(predict, 1)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	saver_net = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
	if os.path.exists(checkpoint_path):
		saver_net.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
	else:
		print("Please trainning data firstly!")
		sess.close()
		exit(-1)
	result = []
	for batch_i in range(math.ceil(test_n/batch_size)):
		images_test = get_next_batch_from_test_path(test_imgs, batch_i, IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size)
		index = sess.run([max_idx_p], feed_dict={X: images_test,k_prob: 1.0, is_training: False})
		result.extend(index[0].tolist())
	sess.close()
	return result

if __name__ == '__main__':
	# 准备识别数据
	all_image = LoadImage("./", train=False)
	test_imgs, test_labels = all_image.images_path, all_image.images_labels
	test_n = all_image.image_n
	test_labels = list(map(eval,test_labels))
	# 预测识别
	result = predict(test_imgs, test_n, config.batch_size,
		  config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.num_classes,
		  config.net_model, config.num_checkpoints, config.checkpoint_path)
	print(result)	
	print(test_labels)
	print("accuracy_score:{}".format(accuracy_score(test_labels,result)))
