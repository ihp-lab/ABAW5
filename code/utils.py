import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from data_multimodal import MyDataset, MyInferenceDataset


def set_seed(seed):
	# Reproducibility
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

	random.seed(seed)
	np.random.seed(seed)


def get_data_loader(csv_file, train, config):
	dataset = MyDataset(csv_file, train, config)
	loader = DataLoader(
					dataset=dataset,
					batch_size=config.batch_size,
					num_workers=config.num_workers,
					shuffle=train,
					collate_fn=dataset.collate_fn,
					drop_last=train)

	return loader


def get_inference_data_loader(csv_file, train, config):
	dataset = MyInferenceDataset(csv_file, train, config)
	loader = DataLoader(
					dataset=dataset,
					batch_size=config.batch_size,
					num_workers=config.num_workers,
					shuffle=train,
					collate_fn=dataset.collate_fn,
					drop_last=train)

	return loader
