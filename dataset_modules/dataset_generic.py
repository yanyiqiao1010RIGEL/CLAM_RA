import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		 ##########
		csv_path = 'dataset_csv/train_hpa_onehot.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		#label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		#label_col = None,
		patient_voting = 'max',
		#######Rigel exp2
		num_classes=28  #
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		#self.label_dict = label_dict
		#self.num_classes = len(set(self.label_dict.values()))
		self.num_classes = num_classes
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		### Rigel changed "label" to "IDH"
		#if not label_col:
		#	label_col = 'label'
		#self.label_col = label_col

		slide_data = pd.read_csv(csv_path)

		if 'IDH' in slide_data.columns:
			slide_data.rename(columns={'IDH': 'label'}, inplace=True)
			self.label_col = 'label'
		elif 'label' in slide_data.columns:
			self.label_col = 'label'
		else:
			raise ValueError("Neither 'IDH' nor 'label' column found in CSV.")
#########Rigel exp2
		#slide_data = self.filter_df(slide_data, filter_dict)
		#slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)
		slide_data['label'] = slide_data['label'].apply(
			lambda x: np.array(list(map(int, x.split(' '))), dtype=np.int32))

		slide_data = self.filter_df(slide_data, filter_dict)
		self.slide_data = slide_data
		print("slide_data.head():",slide_data.head())  # 直接看 DataFrame 解析后的数据

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			#np.random.shuffle(slide_data)
			self.slide_data = self.slide_data.sample(frac=1).reset_index(drop=True)

		#self.slide_data = slide_data

# ### Rigel add one if condition
# 		if self.patient_strat:
# 			self.patient_data_prep(patient_voting)
#
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	# def cls_ids_prep(self):
	# 	# store ids corresponding each class at the patient or case level
	# 	### Rigel add one if condition
	# 	if self.patient_strat:
	# 		self.patient_cls_ids = [[] for i in range(self.num_classes)]
	# 		for i in range(self.num_classes):
	# 			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]
	#
	# 	# store ids corresponding each class at the slide level
	# 	self.slide_cls_ids = [[] for i in range(self.num_classes)]
	# 	for i in range(self.num_classes):
	# 		self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
	def cls_ids_prep(self):
		# 初始化类别计数器
		self.slide_cls_ids = [[] for _ in range(self.num_classes)]
		# print("\nChecking label data structure:")
		# for i in range(min(5, len(self.slide_data))):  # 只打印前5个，防止数据太多
		# 	labels = self.slide_data.iloc[i]['label']
		# 	print(
		# 		f"Index {i}: Type: {type(labels)}, Shape: {labels.shape if isinstance(labels, np.ndarray) else 'N/A'}, Labels: {labels}")

		for i, labels in enumerate(self.slide_data['label']):
			label_indices = np.where(labels == 1)[0]
			#print(f"Index {i}, Labels: {labels}, Label indices: {label_indices}")

			for label_idx in label_indices:
				self.slide_cls_ids[label_idx].append(i)

				# print("\nFinal slide_cls_ids:")
				# for class_idx, sample_indices in enumerate(self.slide_cls_ids):
				# 	print(f"Class {class_idx}: {sample_indices}")


	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	#@staticmethod
	# def df_prep(data, label_dict, ignore, label_col):
	# 	#if label_col != 'label':
	# 	#	data['label'] = data[label_col].copy()
	#
	# 	mask = data['label'].isin(ignore)
	# 	data = data[~mask]
	# 	data.reset_index(drop=True, inplace=True)
	# 	for i in data.index:
	# 		key = data.loc[i, 'label']
	# 		data.at[i, 'label'] = label_dict[key]
	#
	# 	return data
	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		# 如果忽略某些标签，过滤掉相关行
		mask = data[label_col].apply(lambda x: any([int(l) in ignore for l in x.split(' ')]))
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)

		# 确保标签转换为整数列表
		data['label'] = data[label_col].apply(lambda x: np.array(list(map(int, x.split(' '))), dtype=np.int32))

		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		###filter test
		print(f"Filtered dataset size: {len(df)}")
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	# def summarize(self):
	# 	print("label column: {}".format(self.label_col))
	# 	print("label dictionary: {}".format(self.label_dict))
	# 	print("number of classes: {}".format(self.num_classes))
	# 	print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("number of classes: {}".format(self.num_classes))
		print("Slide-LVL; Number of samples registered in each class:")
		class_counts = np.zeros(self.num_classes, dtype=int)
		for labels in self.slide_data['label']:
			for i, is_present in enumerate(labels):  # 遍历独热编码
				if is_present == 1:
					class_counts[i] += 1
		for i, count in enumerate(class_counts):
			print(f'Class {i}: {count} samples')


		# ### Rigel add one if condition
		# if self.patient_strat:
		# 	for i in range(self.num_classes):
		# 		print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
		#
		# for i in range(self.num_classes):
		# 	print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)
		# 统计每个标签的样本索引
		label_indices = {i: [] for i in range(self.num_classes)}
		for idx, labels in enumerate(self.slide_data['label']):
			for i, is_present in enumerate(labels):
				if is_present == 1:
					label_indices[i].append(idx)

		# 统计每个标签的样本数量
		label_counts = {label: len(indices) for label, indices in label_indices.items()}

		# 筛选出样本数 >= 10 的标签
		valid_labels = [label for label, count in label_counts.items() if count >= 10]
		excluded_labels = [label for label, count in label_counts.items() if count < 10]

		print(f"Valid labels for stratification: {valid_labels}")
		print(f"Excluded labels (fewer than 10 samples): {excluded_labels}")

		# 根据每个标签的索引进行分层抽样
		splits = {'train': [], 'val': [], 'test': []}
		# for label, indices in label_indices.items():
		# 	np.random.seed(self.seed)
		# 	np.random.shuffle(indices)
		for label in valid_labels:
			indices = label_indices[label]
			np.random.seed(self.seed)
			np.random.shuffle(indices)

			num_train = int(len(indices) * label_frac)
			num_val = val_num[0] if len(indices) >= sum(val_num) else len(indices) // 2
			num_test = test_num[0] if len(indices) >= sum(test_num) else len(indices) - num_train - num_val

			train_indices = indices[:num_train]
			val_indices = indices[num_train:num_train + num_val]
			test_indices = indices[num_train + num_val:num_train + num_val + num_test]

			splits['train'].extend(train_indices)
			splits['val'].extend(val_indices)
			splits['test'].extend(test_indices)

		# 对于被排除的标签，直接将样本加入训练集
		for label in excluded_labels:
			splits['train'].extend(label_indices[label])

		# 去重并保存
		splits = {key: list(set(indices)) for key, indices in splits.items()}
		self.train_ids, self.val_ids, self.test_ids = splits['train'], splits['val'], splits['test']

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		self.train_ids, self.val_ids, self.test_ids = ids

		# 确保多标签样本的完整性
		self.train_ids = list(set(self.train_ids))
		self.val_ids = list(set(self.val_ids))
		self.test_ids = list(set(self.test_ids))

		# if self.patient_strat:
		# 	slide_ids = [[] for i in range(len(ids))]
		#
		# 	for split in range(len(ids)):
		# 		for idx in ids[split]:
		# 			case_id = self.patient_data['case_id'][idx]
		# 			slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
		# 			slide_ids[split].extend(slide_indices)
		#
		# 	self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
		#
		# else:
		# 	self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		#######Rigel test
		print(f"Columns in all_splits: {all_splits.columns}")
		print(f"Content of '{split_key}' column: {all_splits[split_key].head()}")
		print(f"slide_id dtype: {self.slide_data['slide_id'].dtype}")

		# 检查所有的 slide_id 是否在 train、val 和 test 列中
		train_ids_in_slide_data = all_splits['train'].isin(self.slide_data['slide_id']).sum()
		val_ids_in_slide_data = all_splits['val'].isin(self.slide_data['slide_id']).sum()
		test_ids_in_slide_data = all_splits['test'].isin(self.slide_data['slide_id']).sum()

		print(f"Number of train ids found in slide_data: {train_ids_in_slide_data}")
		print(f"Number of val ids found in slide_data: {val_ids_in_slide_data}")
		print(f"Number of test ids found in slide_data: {test_ids_in_slide_data}")

		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			print(f"Split for {split_key} contains {len(df_slice)} items.")


			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):


		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
			
			else:
				test_split = None
			
		
		else:
			assert csv_path
			print(f"CSV file path: {csv_path}")
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			#all_splits = pd.read_csv(csv_path, dtype={'slideid': str}, on_bad_lines='skip',  skip_blank_lines=True)
			#print("ggggggget_split_from_df all splits:", all_splits.iloc[0])
			train_split = self.get_split_from_df(all_splits, 'train')
			print("geeeeeeeet_split_from_df train splits:", train_split[0])
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')

			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			# index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			index = [i for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		# unique, counts = np.unique(labels, return_counts=True)
		# for u in range(len(unique)):
		# 	print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
		# 	if return_descriptor:
		# 		df.loc[index[u], 'train'] = counts[u]
		label_count = np.zeros(self.num_classes, dtype=int)
		for i, label in enumerate(labels):
			for l in range(self.num_classes):  # 遍历所有类别
				if label[l] == 1:  # 如果该类别的标签为 1，才计数
					label_count[l] += 1

		# 打印每个类别的样本数，并确保计数与类别正确对应
		for i in range(self.num_classes):
			count = label_count[i]
			print(f'Number of samples in class {i}: {count}')

			if return_descriptor:
				df.loc[i, 'train'] = count  # 如果需要在 dataframe 中记录
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		# unique, counts = np.unique(labels, return_counts=True)
		# for u in range(len(unique)):
		# 	print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
		# 	if return_descriptor:
		# 		df.loc[index[u], 'val'] = counts[u]
		label_count = np.zeros(self.num_classes, dtype=int)
		for i, label in enumerate(labels):
			for l in range(self.num_classes):  # 遍历所有类别
				if label[l] == 1:  # 如果该类别的标签为 1，才计数
					label_count[l] += 1

		# 打印每个类别的样本数，并确保计数与类别正确对应
		for i in range(self.num_classes):
			count = label_count[i]
			print(f'Number of samples in class {i}: {count}')

			if return_descriptor:
				df.loc[i, 'val'] = count  # 如果需要在 dataframe 中记录


		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		# unique, counts = np.unique(labels, return_counts=True)
		# for u in range(len(unique)):
		# 	print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
		# 	if return_descriptor:
		# 		df.loc[index[u], 'test'] = counts[u]
		label_count = np.zeros(self.num_classes, dtype=int)
		for i, label in enumerate(labels):
			for l in range(self.num_classes):  # 遍历所有类别
				if label[l] == 1:  # 如果该类别的标签为 1，才计数
					label_count[l] += 1

		# 打印每个类别的样本数，并确保计数与类别正确对应
		for i in range(self.num_classes):
			count = label_count[i]
			print(f'Number of samples in class {i}: {count}')

			if return_descriptor:
				df.loc[i, 'test'] = count  # 如果需要在 dataframe 中记录



		#assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		#assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		#assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0
		# 确保train_ids和test_ids没有交集
		train_ids_set = set(self.train_ids)
		test_ids_set = set(self.test_ids)
		val_ids_set = set(self.val_ids)

		# 打印交集
		print(f"Intersection between train and test: {train_ids_set & test_ids_set}")
		print(f"Intersection between train and val: {train_ids_set & val_ids_set}")
		print(f"Intersection between val and test: {val_ids_set & test_ids_set}")
		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir,
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		print(f"Data directory: {data_dir}")
		self.data_dir = data_dir
		self.use_h5 = True

		print(f"Initializing dataset with {len(self.slide_data)} entries")

		### slide_id fliter
		self.filter_invalid_h5_files()

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def filter_invalid_h5_files(self):

		existing_files = set([f.split('.h5')[0] for f in os.listdir(self.data_dir)])

		# filter
		initial_len = len(self.slide_data)
		self.slide_data = self.slide_data[self.slide_data['slide_id'].isin(existing_files)]
		self.slide_data.reset_index(drop=True, inplace=True)
		filtered_len = len(self.slide_data)
		print(f"Filtered initial_len={initial_len}")
		print(f"Filtered filtered_len={filtered_len}")
		print(f"Filtered {initial_len - filtered_len} entries without corresponding .h5 files.")

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		# if type(self.data_dir) == dict:
		# 	source = self.slide_data['source'][idx]
		# 	data_dir = self.data_dir[source]
		# else:
		# 	data_dir = self.data_dir
		#####Rigel exp2
		print(f"Inside __getitem__ - Index {idx}")
		print(f"  - slide_id: {slide_id}")
		print(f"  - Label Type: {type(label)}, Label: {label}")
		if isinstance(label, np.ndarray):
			print(f"  🔹 Label is a numpy array, shape: {label.shape}")
		elif isinstance(label, torch.Tensor):
			print(f"  🔴 Label has already become a tensor!")

		one_hot_label = np.zeros(self.num_classes, dtype=np.float32)
		one_hot_label[label] = 1.0  # 将对应索引位置设置为 1

		if self.use_h5:
			full_path = os.path.join(self.data_dir, f'{slide_id}.h5')
			with h5py.File(full_path, 'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, torch.tensor(one_hot_label), coords

		else:
			if self.data_dir:
				full_path = os.path.join(self.data_dir, f'{slide_id}.pt')
				features = torch.load(full_path)
				return features, torch.tensor(one_hot_label)

			return slide_id, torch.tensor(one_hot_label)


		# if not self.use_h5:
		# 	if self.data_dir:
		# 		full_path = os.path.join(data_dir, '{}.pt'.format(slide_id))
		# 		features = torch.load(full_path)
		# 		return features, label
		#
		# 	else:
		# 		return slide_id, label
		#
		# else:
		# 	full_path = os.path.join(data_dir, '{}.h5'.format(slide_id))
		#
		# 	with h5py.File(full_path,'r') as hdf5_file:
		# 		features = hdf5_file['features'][:]
		# 		coords = hdf5_file['coords'][:]
		#
		# 	features = torch.from_numpy(features)
		# 	return features, label, coords


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=28):
		self.use_h5 = True
		self.slide_data = slide_data
		self.data_dir = data_dir
		#print(f"datadir after init:{self.data_dir}")
		self.num_classes = num_classes
		#print(f"Num class={self.num_classes}")
		self.slide_cls_ids = [[] for i in range(self.num_classes)]

		for idx, labels in enumerate(self.slide_data['label']):
			print(f"Index {idx} - Labellllllll: {labels} - Type: {type(labels)}")
			#print(f"Label Shape: {labels.shape if isinstance(labels, np.ndarray) else 'Not an array'}")

			for i, is_present in enumerate(labels):
				#print(f"Checking label[{i}]: {is_present}")
				if is_present == 1:
					self.slide_cls_ids[i].append(idx)



	def __len__(self):
		return len(self.slide_data)
		


