import os
import random
import numpy as np
import pandas as pd
import pickle5 as pickle
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence


def transcript_tokens_reverse(utt_tokens, start_word, end_word):
	# return start and end index of utt_tokens that bounded by the start_word and end_word
	# each token can be sub-word
	# start, end = 1, len(utt_tokens)-1

	start, end = None, None
	for i in range(len(utt_tokens)):
		cur_token = utt_tokens[i]
		if start_word.startswith(cur_token):
			start = i
		if start is not None and end_word.endswith(cur_token):
			end = i
			return start, end
	if start is None or end is None or start > end:
		return 1, len(utt_tokens)-1

	return start, end


class image_train(object):
	def __init__(self, img_size=256, crop_size=224):
		self.img_size = img_size
		self.crop_size = crop_size

	def __call__(self, img):
		transform = transforms.Compose([
			transforms.Resize(self.img_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5],
								 std=[0.5, 0.5, 0.5])
		])
		img = transform(img)

		return img


class image_test(object):
	def __init__(self, img_size=256, crop_size=224):
		self.img_size = img_size
		self.crop_size = crop_size

	def __call__(self, img):
		transform = transforms.Compose([
			transforms.Resize(self.img_size),
			transforms.CenterCrop(self.crop_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5],
								 std=[0.5, 0.5, 0.5])
		])
		img = transform(img)

		return img


class MyDataset(data.Dataset):
	def __init__(self, csv_file, train, config):
		self.config = config
		self.csv_file = csv_file

		self.data_root = config.data_root
		self.img_size = config.image_size
		self.crop_size = config.crop_size
		self.train = train
		if self.train:
			self.transform = image_train(img_size=self.img_size, crop_size=self.crop_size)
		else:
			self.transform = image_test(img_size=self.img_size, crop_size=self.crop_size)

		self.file_list = pd.read_csv(csv_file)
		self.images = self.file_list['image_path']
		self.labels = [
			self.file_list['au1'],
			self.file_list['au2'],
			self.file_list['au4'],
			self.file_list['au6'],
			self.file_list['au7'],
			self.file_list['au10'],
			self.file_list['au12'],
			self.file_list['au15'],
			self.file_list['au23'],
			self.file_list['au24'],
			self.file_list['au25'],
			self.file_list['au26']
		]
		self.num_labels = len(self.labels)
		video_file_ids = list(set([x.split('/')[-2] for x in self.images]))
		for x in video_file_ids:
			if '_left' in x:
				video_file_ids.append(x.replace('_left', ''))
			elif '_right' in x:
				video_file_ids.append(x.replace('_right', ''))

		self.audio_dict = {}
		for file in os.listdir(self.config.audio_feat_path):
			if file.split('.')[0] in video_file_ids:
				self.audio_dict[file.split('.')[0]] = np.load(os.path.join(self.config.audio_feat_path, file))

		self.text_timestamp_dict = {}
		for file in os.listdir(self.config.text_timestamp_path):
			if file.split('.')[0] in video_file_ids:
				try:
					self.text_timestamp_dict[file.split('.')[0]] = pd.read_csv(os.path.join(self.config.text_timestamp_path, file), header=None).values
				except:
					self.text_timestamp_dict[file.split('.')[0]] = []
					continue

		with open(self.config.text_feat_path, 'rb') as f:
			self.text_feature_dict = pickle.load(f)

		video_fps_info = pd.read_csv(self.config.video_fps_info_path, header=None).values
		self.video_fps_dict = {}
		for (vid_id, vid_fps) in video_fps_info:
			self.video_fps_dict[vid_id] = vid_fps

		self.context_sz = 2 # seconds
		self.hubert_fps = 50
		self.audio_context_sz = self.context_sz * self.hubert_fps

	def data_augmentation(self, image, flip, crop_size, offset_x, offset_y):
		image = image[:,offset_x:offset_x+crop_size,offset_y:offset_y+crop_size]
		if flip:
			image = torch.flip(image, [2])

		return image

	def pil_loader(self, path):
		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

	def __getitem__(self, index):
		ori_image_path = os.path.join(self.config.data_root, 'cropped_aligned', self.images[index].replace('_out', ''))
		super_res_image_path = os.path.join(self.config.data_root, 'super-res-aligned', self.images[index])

		ori_image = self.pil_loader(ori_image_path)
		ori_image = self.transform(ori_image)
		if self.train:
			offset_y = random.randint(0, self.img_size - self.crop_size)
			offset_x = random.randint(0, self.img_size - self.crop_size)
			flip = random.randint(0, 1)
			ori_image = self.data_augmentation(ori_image, flip, self.crop_size, offset_x, offset_y)

		label = []
		for i in range(self.num_labels):
			label.append(int(self.labels[i][index]))
		label = torch.FloatTensor(label)

		# Extract gh, audio, and text features
		image_path = ori_image_path
		cur_frame_num = int(os.path.split(image_path)[-1].split('.')[0])
		cur_file_id = image_path.split('/')[-2]
		cur_file_id = cur_file_id.replace('_left', '') if '_left' in cur_file_id else cur_file_id
		cur_file_id = cur_file_id.replace('_right', '') if '_right' in cur_file_id else cur_file_id
		cur_fps = self.video_fps_dict[cur_file_id]
		cur_timestamp = cur_frame_num/cur_fps

		gh_feat_path = super_res_image_path.replace('super-res-aligned', 'gh-features')[:-4]+'.npy'
		if os.path.exists(gh_feat_path):
			gh_feat = np.load(gh_feat_path)
		else:
			gh_feat = np.zeros((14,1024))
		gh_feat = torch.from_numpy(gh_feat).float()

		cur_start_frame, cur_end_frame = int(cur_frame_num-self.context_sz*cur_fps), int(cur_frame_num+self.context_sz*cur_fps)
		cur_context_gh_feats = []
		cur_gh_dir = os.path.split(gh_feat_path)[0]

		def sample_frames(start_frame, stop_frame, num_frames=8):
			frame_list = range(start_frame, stop_frame+1)
			return [frame_list[i] for i in np.linspace(0, len(frame_list)-1, num_frames).astype('int')]

		self.num_frames = self.config.num_frames
		for i in sample_frames(cur_start_frame, cur_end_frame, self.num_frames):
			cur_context_frame_gh_path = os.path.join(cur_gh_dir, '{}_out.npy'.format(str(i).zfill(5)))
			if os.path.exists(cur_context_frame_gh_path):
				cur_context_gh_feats.append(np.load(cur_context_frame_gh_path))
		if len(cur_context_gh_feats) == 0:
			cur_context_gh_feats = np.zeros((1,14,1024))
			cur_context_gh_feats[0] = gh_feat
		cur_context_gh_feats = np.array(cur_context_gh_feats)

		cur_audio_feat = self.audio_dict[cur_file_id]
		t_hubert = int(cur_timestamp*self.hubert_fps)
		cur_audio_feat = cur_audio_feat[int(t_hubert-self.audio_context_sz) : int(t_hubert+self.audio_context_sz)]

		cur_transcript = self.text_feature_dict[cur_file_id]
		cur_text_timestamp = self.text_timestamp_dict[cur_file_id]
		cur_start_timestamp, cur_end_timestamp = cur_timestamp-self.context_sz, cur_timestamp+self.context_sz
		
		transcript_ids, start_word, end_word = [], None, None
		cur_text_timestamp = self.text_timestamp_dict[cur_file_id]
		if len(cur_text_timestamp) == 0:
			text_features = np.zeros((1,1024))
		else:
			for row in cur_text_timestamp:
				cur_word, cur_utt_id, cur_word_timestamp = row
				if cur_word_timestamp >= cur_start_timestamp and start_word is None:
					start_word = cur_word
					transcript_ids.append(cur_utt_id)
				elif cur_start_timestamp <= cur_word_timestamp <= cur_end_timestamp:
					transcript_ids.append(cur_utt_id)
					end_word = cur_word
				elif cur_word_timestamp > cur_end_timestamp:
					break

		if len(transcript_ids) == 0 or len(cur_transcript) == 0:
			text_features = np.zeros((1,1024))
		elif len(transcript_ids) == 1:
			end_word = start_word
			cur_utt_id = transcript_ids[0]
			utt_start_idx, utt_end_idx = transcript_tokens_reverse(cur_transcript[cur_utt_id][0], start_word.lower(), end_word.lower())
			text_features = cur_transcript[cur_utt_id][1][utt_start_idx:utt_end_idx+1]
		else:
			cur_utt_id = transcript_ids[0]
			utt_start_idx, utt_end_idx = transcript_tokens_reverse(cur_transcript[cur_utt_id][0], start_word.lower(), end_word.lower())	
			text_features = cur_transcript[cur_utt_id][1][utt_start_idx:utt_end_idx+1]

		return ori_image, gh_feat, torch.FloatTensor(cur_context_gh_feats), torch.FloatTensor(cur_audio_feat), torch.FloatTensor(text_features), label

	def collate_fn(self, data):
		ori_images, gh_feats, cur_context_gh_feats, audio_feats, text_feats, labels = zip(*data)

		ori_images = torch.stack(ori_images)
		gh_feats = torch.stack(gh_feats)
		cur_context_gh_feats = pad_sequence(cur_context_gh_feats, batch_first=True)
		audio_feats = pad_sequence(audio_feats, batch_first=True)
		text_feats = pad_sequence(text_feats, batch_first=True)
		labels = torch.stack(labels)

		return ori_images, gh_feats, cur_context_gh_feats, audio_feats, text_feats, labels

	def __len__(self):
		return len(self.images)
