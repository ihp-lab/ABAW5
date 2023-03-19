import os
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import get_data_loader
from models.swin import SwinV0, SwinV1, SwinV2


class solver_multimodal(nn.Module):
	def __init__(self, config):
		super(solver_multimodal, self).__init__()
		self.config = config
		self.init()


	def init(self):
		# Setup number of labels
		self.num_labels = self.config.num_labels

		# Initiate data loaders
		self.get_data_loaders()

		# Initiate the networks
		if self.config.model == 'v0':
			if self.config.model_name == 'none':
				self.config.model_name = 'v0'
			self.model = SwinV0(self.config).cuda()
		elif self.config.model == 'v1':
			if self.config.model_name == 'none':
				self.config.model_name = 'v1'
			self.model = SwinV1(self.config).cuda()
		elif self.config.model == 'v2':
			if self.config.model_name == 'none':
				self.config.model_name = 'v2'
			self.model = SwinV2(self.config).cuda()

		# Setup the optimizers and loss function
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
		self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.config.when, factor=0.8, verbose=False)
		self.criterion = nn.BCEWithLogitsLoss()
  
		self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor(
			[1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]).cuda())
		self.multi_label_loss = torch.nn.MultiLabelSoftMarginLoss(weight=torch.Tensor(
			[1, 2, 1, 1, 1, 1, 1, 6, 6, 6, 1, 2]).cuda())

		# Setup AU index
		self.aus = [1,2,4,6,7,10,12,15,23,24,25,26]

		# Select the best ckpt
		self.best_val_metric = 0.


	def get_data_loaders(self):
		train_csv = os.path.join('../labels/train.csv')
		val_csv = os.path.join('../labels/val.csv')
		test_csv = os.path.join('../labels/test.csv')

		self.train_loader = get_data_loader(train_csv, True, self.config)
		self.val_loader = get_data_loader(val_csv, False, self.config)
		self.test_loader = get_data_loader(test_csv, False, self.config)


	def train_model(self, train_loader):
		self.train()
		total_loss, total_sample = 0., 0
		self.config.interval = min(self.config.interval, len(train_loader))

		for i, (ori_images, gh_feats, cur_context_gh_feats, audio_feats, text_feats, labels) in enumerate(tqdm(train_loader, total=self.config.interval)):
			if i >= self.config.interval:
				break
			ori_images, gh_feats, cur_context_gh_feats, labels = ori_images.cuda(), gh_feats.cuda(), cur_context_gh_feats.cuda(), labels.cuda()
			audio_feats, text_feats = audio_feats.cuda(), text_feats.cuda()

			batch_size = ori_images.shape[0]
			self.optimizer.zero_grad()

			preds = self.model(ori_images, gh_feats, cur_context_gh_feats, audio_feats, text_feats)
			if self.config.loss == 'unweighted':
				loss = self.criterion(preds.reshape(-1), labels.reshape(-1))
			elif self.config.loss == 'weighted':
				loss_of_bce = self.bce_loss(preds, labels).mean()
				loss_of_multi_label = self.multi_label_loss(preds, labels)
				loss = loss_of_bce + loss_of_multi_label

			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
			self.optimizer.step()

			total_loss += loss.item()*batch_size
			total_sample += batch_size

		avg_loss = total_loss / total_sample

		return avg_loss


	def val_model(self, val_loader):
		val_loss, val_f1 = self.test_model(val_loader)
		self.save_best_ckpt(sum(val_f1)/len(val_f1))
		self.scheduler.step(val_loss)

		return val_loss, val_f1


	def test_model(self, test_loader):
		with torch.no_grad():
			self.eval()
			total_loss, total_sample = 0., 0
			pred_list, gt_list, f1_list = [], [], []
			for (ori_images, gh_feats, cur_context_gh_feats, audio_feats, text_feats, labels) in tqdm(test_loader):
				ori_images, gh_feats, cur_context_gh_feats, labels = ori_images.cuda(), gh_feats.cuda(), cur_context_gh_feats.cuda(), labels.cuda()
				audio_feats, text_feats = audio_feats.cuda(), text_feats.cuda()

				batch_size = ori_images.shape[0]

				preds = self.model(ori_images, gh_feats, cur_context_gh_feats, audio_feats, text_feats)
				if self.config.loss == 'unweighted':
					loss = self.criterion(preds.reshape(-1), labels.reshape(-1))
				elif self.config.loss == 'weighted':
					loss_of_bce = self.bce_loss(preds, labels).mean()
					loss_of_multi_label = self.multi_label_loss(preds, labels)
					loss = loss_of_bce + loss_of_multi_label

				total_loss += loss.item()*batch_size
				total_sample += batch_size

				preds = (preds >= self.config.threshold).int()

				pred_list.append(preds)
				gt_list.append(labels)

			avg_loss = total_loss / total_sample

			pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
			gt_list = torch.cat(gt_list, dim=0).detach().cpu().numpy()

			for i in range(self.num_labels):
				f1_list.append(100.0*f1_score(gt_list[:, i], pred_list[:, i]))
	
			return avg_loss, f1_list


	def print_metric(self, f1_list, prefix):
		print('{} avg F1: {:.2f}'.format(prefix, sum(f1_list)/len(f1_list)))
		for i in range(len(self.aus)):
			print('AU {}: {:.2f}'.format(self.aus[i], f1_list[i]), end=' ')
		print('')


	def load_best_ckpt(self):
		ckpt_name = os.path.join(self.config.ckpt_path, self.config.model_name+'.pt')
		checkpoints = torch.load(ckpt_name)['model']
		self.model.load_state_dict(checkpoints, strict=True)


	def save_best_ckpt(self, val_metric):
		def update_metric(val_metric):
			if val_metric > self.best_val_metric:
				self.best_val_metric = val_metric
				return True
			return False

		if update_metric(val_metric):
			os.makedirs(os.path.join(self.config.ckpt_path), exist_ok=True)
			ckpt_name = os.path.join(self.config.ckpt_path, self.config.model_name+'.pt')
			torch.save({'model': self.model.state_dict()}, ckpt_name)
			print('save to:', ckpt_name)


	def run(self):
		best_val_f1 = 0.

		# Load pretrain weights
		if not self.config.pretrain == 'none':
			checkpoints = torch.load(self.config.pretrain)['model']
			self.model.load_state_dict(checkpoints, strict=True)

			_, val_f1 = self.val_model(self.val_loader)
			self.print_metric(val_f1, 'Pretrain')
			best_val_f1 = sum(val_f1)/len(val_f1)

		patience = self.config.patience
		for epochs in range(1, self.config.num_epochs+1):
			print('Epoch: %d/%d' % (epochs, self.config.num_epochs))

			# Train model
			train_loss = self.train_model(self.train_loader)
			print('Training loss: {:.6f}'.format(train_loss))

			# Validate model
			_, val_f1 = self.val_model(self.val_loader)
			self.print_metric(val_f1, 'Val')

			if sum(val_f1)/len(val_f1) > best_val_f1:
				patience = self.config.patience
				best_val_f1 = sum(val_f1)/len(val_f1)
			else:
				patience -= 1
				if patience == 0:
					break

		# Test model
		self.load_best_ckpt()
		_, test_f1 = self.test_model(self.test_loader)
		self.print_metric(test_f1, 'Test')
