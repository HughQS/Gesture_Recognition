import numpy as np
import cv2
import data_aug




class LoadImage(object):
	
	def __init__(self, root, transforms=None,train=True):
		self.train = train
		self.images_root = root
		self._read_txt_file()
	
	def _read_txt_file(self):
		self.images_path = []
		self.images_labels = []

		if self.train:
			txt_file = self.images_root + "./images/train.txt"
		else:
			txt_file = self.images_root + "./images/test.txt"

		with open(txt_file, 'r') as f:
			lines = f.readlines()
			for line in lines:
				item = line.strip().split(' ')
				self.images_path.append(item[0])
				self.images_labels.append(item[1])
		self.num_classes = len(set(self.images_labels))
		self.image_n = len(self.images_path)
		
	def _shuffle_train_data(self):
		"""数据集转为多维数组并打乱"""
		index = [i for i in range(len(self.images_path))]
		np.random.shuffle(index)
		self.images_path = np.asarray(self.images_path)
		self.images_labels = np.asarray(self.images_labels)
		self.images_path = self.images_path[index]
		self.images_labels = self.images_labels[index]

	def gen_train_valid_image(self, train_rate=0.9):
			"""生成训练和验证数据
			Args:
				train_rate: 训练数据占比
			"""
			self.train_rate = train_rate
			# 打乱数据集
			self._shuffle_train_data()
			self.train_n = int(self.image_n * self.train_rate)
			self.valid_n = int(self.image_n * (1 - self.train_rate))
			return self.images_path[0:self.train_n], self.images_labels[0:self.train_n],\
			self.images_path[self.train_n:self.image_n], self.images_labels[self.train_n:self.image_n]

def shuffle_train_data(train_imgs, train_labels):
	"""打乱数据"""
	index = [i for i in range(len(train_imgs))]
	np.random.shuffle(index)
	train_imgs = np.asarray(train_imgs)
	train_labels = np.asarray(train_labels)
	train_imgs = train_imgs[index]
	train_labels = train_labels[index]
	return train_imgs, train_labels

def get_next_batch_from_path(image_path, image_labels, pointer, IMAGE_HEIGHT=64, IMAGE_WIDTH=64, batch_size=64, is_train=True):
	"""批量生成图像"""
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT,IMAGE_WIDTH,3])
	num_classes = len(image_labels[0])
	batch_y = np.zeros([batch_size, num_classes]) 
	for i in range(batch_size):
		image = cv2.imread(image_path[i+pointer*batch_size])
		image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
		if is_train:
			img_aug = data_aug.data_aug(image)
			image = img_aug.get_aug_img()
		image = image / 255.0
		image = image - 0.5
		image = image * 2
		batch_x[i,:,:,:] = image
		batch_y[i] = image_labels[i+pointer*batch_size]
	return batch_x, batch_y


def get_next_batch_from_test_path(image_path, pointer, IMAGE_HEIGHT=64, IMAGE_WIDTH=64, batch_size=64):
	"""批量生成图像"""
	if ((pointer+1)*batch_size) < len(image_path):
		tmp_batch_size = batch_size
	else:
		tmp_batch_size = len(image_path)- pointer*batch_size
		print(str(tmp_batch_size+pointer*batch_size))
		print(tmp_batch_size)
	batch_x = np.zeros([tmp_batch_size, IMAGE_HEIGHT,IMAGE_WIDTH,3])
	for i in range(tmp_batch_size):
		image = cv2.imread(image_path[i+pointer*batch_size])
		image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
		image = image / 255.0
		image = image - 0.5
		image = image * 2
		
		batch_x[i,:,:,:] = image
	return batch_x
		
if __name__ == '__main__':
	xx = LoadImage("./")
	print(xx.images_path[0])
	print(xx.images_labels[0])
	print(len(xx.images_path))