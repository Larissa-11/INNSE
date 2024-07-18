import os.path as osp
import torch.utils.data as data
import data.util as util


class VideoTestDataset(data.Dataset):
	"""
	A video test dataset. Support:
	Vid4
	REDS4
	Vimeo90K-Test

	no need to prepare LMDB files
	"""

	def __init__(self, opt):
		super(VideoTestDataset, self).__init__()
		self.opt = opt
		self.cache_data = opt['cache_data']
		self.half_N_frames = opt['N_frames'] // 2
		self.GT_root = opt['dataroot_GT']
		self.data_type = self.opt['data_type']
		self.data_info = {'path_GT': [], 'folder': [], 'idx': [], 'border': []}

		if self.data_type == 'lmdb':
			raise ValueError('No need to use LMDB during validation/test.')
		#### Generate data info and cache data
		self.imgs_GT =  {}
		if opt['name'].lower() in ['vid4', 'reds4', 'decodehr']:
			subfolders_GT = util.glob_file_list(self.GT_root)
			for subfolder_GT in zip(subfolders_GT):
				subfolder_name = osp.basename(subfolder_GT[0])
				# img_paths_LQ = util.glob_file_list(subfolder_LQ)
				img_paths_GT = util.glob_file_list(subfolder_GT[0])
				max_idx = len(img_paths_GT)
				if opt['pred_interval'] == 0:
					self.data_info['path_GT'].extend(img_paths_GT)
					self.data_info['folder'].extend([subfolder_name] * max_idx)
					for i in range(max_idx):
						self.data_info['idx'].append('{}/{}'.format(i, max_idx))
				elif opt['pred_interval'] < 0:
					GTs = []
					GTs.append(img_paths_GT)
					self.data_info['path_GT'].extend(GTs)
					self.data_info['folder'].extend([subfolder_name])
					self.data_info['idx'].append('{}/{}'.format(1, 1))
				else:
					self.pred_interval = opt['pred_interval']
					GTs = []
					if max_idx % self.pred_interval == 1 or max_idx % self.pred_interval == 0:
						num_clip = max_idx // self.pred_interval
					else:
						num_clip = max_idx // self.pred_interval + 1

					for i in range(num_clip):
						if i != max_idx // self.pred_interval:
							GTs.append(img_paths_GT[i * self.pred_interval: (i+1) * self.pred_interval + 1])
						else:
							GTs.append(img_paths_GT[i * self.pred_interval:])

					self.data_info['path_GT'].extend(GTs)
					self.data_info['folder'].extend([subfolder_name] * num_clip)
					for i in range(max_idx // self.pred_interval + 1):
						self.data_info['idx'].append('{}/{}'.format(i, num_clip))

				if self.cache_data:
					self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)
		elif opt['name'].lower() in ['vimeo90k-test']:
			pass  # TODO
		else:
			raise ValueError(
				'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

	def __getitem__(self, index):
		path_GT = self.data_info['path_GT'][index]
		folder = self.data_info['folder'][index]
		idx, max_idx = self.data_info['idx'][index].split('/')
		idx, max_idx = int(idx), int(max_idx)

		if self.cache_data:
			img_GT = self.imgs_GT[folder][idx]
		else:
			img_GT = util.read_img_seq(path_GT)

		return {
			'GT': img_GT,
			'folder': folder,
			'idx': self.data_info['idx'][index],
			'GT_path': path_GT
		}

	def __len__(self):
		return len(self.data_info['path_GT'])
