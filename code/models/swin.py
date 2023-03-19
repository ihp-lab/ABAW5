import torch
import torch.nn as nn

from models.swin_transformer import swin_transformer_base, swin_transformer_tiny


class SwinTransformerBase(nn.Module):
	def __init__(self, opts):
		super(SwinTransformerBase, self).__init__()

		self.encoder = swin_transformer_base()
		self.classifier = nn.Sequential(
			nn.Linear(49*1024, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))

	def forward(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels = self.classifier(features)

		return labels


class SwinTransformerTiny(nn.Module):
	def __init__(self, opts):
		super(SwinTransformerTiny, self).__init__()

		self.encoder = swin_transformer_tiny()
		self.classifier = nn.Sequential(
			nn.Linear(49*768, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))

	def forward(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels = self.classifier(features)

		return labels


class SwinV0(nn.Module):
	def __init__(self, opts):
		super(SwinV0, self).__init__()

		self.vision_encoder_1 = swin_transformer_tiny()

		self.static_proj_1 = nn.Linear(49*768, 512)

		self.embed_dim = 64
		self.audio_dim = 1280
		self.text_dim = 1024
		self.num_layers = 2
		self.audio_proj = nn.Linear(self.audio_dim, self.embed_dim)
		self.audio_encoder = nn.GRU(
								input_size=self.embed_dim,
								hidden_size=self.embed_dim,
								num_layers=self.num_layers,
								dropout=opts.dropout,
								batch_first=True,
								bidirectional=True
							)

		self.text_proj = nn.Linear(self.text_dim, self.embed_dim)
		self.text_encoder = nn.GRU(
	  							input_size=self.embed_dim,
			 					hidden_size=self.embed_dim,
				  				num_layers=self.num_layers,
					  			dropout=opts.dropout,
								batch_first=True,
								bidirectional=True
			  				)

		self.classifier = nn.Sequential(
			nn.Linear(512+128*2, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))

	def forward(self, ori_images, gh_feats, cur_context_gh_feats, audio_feats, text_feats):
		batch_size = ori_images.shape[0]
		ori_visual_reps = self.vision_encoder_1(ori_images).reshape(batch_size, -1)
		ori_visual_reps = self.static_proj_1(ori_visual_reps)

		audio_feats = self.audio_proj(audio_feats)
		audio_outputs, _ = self.audio_encoder(audio_feats) # BxTx128
		audio_reps = audio_outputs[:,-1,:]

		text_feats = self.text_proj(text_feats)
		text_outputs, _ = self.text_encoder(text_feats) # BxTx128
		text_reps = text_outputs[:,-1,:]

		labels = self.classifier(torch.cat([ori_visual_reps, audio_reps, text_reps], dim=1))

		return labels


class SwinV1(nn.Module):
	def __init__(self, opts):
		super(SwinV1, self).__init__()

		self.vision_encoder_1 = swin_transformer_tiny()
		# self.vision_encoder_2 = swin_transformer_tiny()

		self.static_proj_1 = nn.Linear(49*768, 512)
		# self.static_proj_2 = nn.Linear(49*768, 512)
		self.static_proj_3 = nn.Linear(14*1024, 128)

		self.embed_dim = 64
		self.audio_dim = 1280
		self.text_dim = 1024
		self.num_layers = 2
		self.audio_proj = nn.Linear(self.audio_dim, self.embed_dim)
		self.audio_encoder = nn.GRU(
								input_size=self.embed_dim,
								hidden_size=self.embed_dim,
								num_layers=self.num_layers,
								dropout=opts.dropout,
								batch_first=True,
								bidirectional=True
							)

		self.text_proj = nn.Linear(self.text_dim, self.embed_dim)
		self.text_encoder = nn.GRU(
	  							input_size=self.embed_dim,
			 					hidden_size=self.embed_dim,
				  				num_layers=self.num_layers,
					  			dropout=opts.dropout,
								batch_first=True,
								bidirectional=True
			  				)

		self.classifier = nn.Sequential(
			nn.Linear(512+128*3, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))

	def forward(self, ori_images, gh_feats, cur_context_gh_feats, audio_feats, text_feats):
		batch_size = ori_images.shape[0]
		ori_visual_reps = self.vision_encoder_1(ori_images).reshape(batch_size, -1)
		ori_visual_reps = self.static_proj_1(ori_visual_reps)
		# super_res_visual_reps = self.vision_encoder_2(super_res_images).reshape(batch_size, -1)
		# super_res_visual_reps = self.static_proj_2(super_res_visual_reps)

		gh_feats = self.static_proj_3(gh_feats.reshape(batch_size, -1))

		audio_feats = self.audio_proj(audio_feats)
		audio_outputs, _ = self.audio_encoder(audio_feats) # BxTx128
		audio_reps = audio_outputs[:,-1,:]

		text_feats = self.text_proj(text_feats)
		text_outputs, _ = self.text_encoder(text_feats) # BxTx128
		text_reps = text_outputs[:,-1,:]

		labels = self.classifier(torch.cat([ori_visual_reps, gh_feats, audio_reps, text_reps], dim=1))

		return labels


class SwinV2(nn.Module):
	def __init__(self, opts):
		super(SwinV2, self).__init__()

		self.vision_encoder_1 = swin_transformer_tiny()
		# self.vision_encoder_2 = swin_transformer_tiny()

		self.static_proj_1 = nn.Linear(49*768, 512)
		# self.static_proj_2 = nn.Linear(49*768, 512)
		self.static_proj_3 = nn.Linear(14*1024, 128)

		self.embed_dim = 64
		self.audio_dim = 1280
		self.text_dim = 1024
		self.num_layers = 2
		self.num_frames = opts.num_frames

		self.gh_feat_proj = nn.Linear(14*1024, self.embed_dim)
		self.gh_feat_encoder = nn.GRU(
								input_size=self.embed_dim,
								hidden_size=self.embed_dim,
								num_layers=self.num_layers,
								dropout=opts.dropout,
								batch_first=True,
								bidirectional=True
							)

		self.audio_proj = nn.Linear(self.audio_dim, self.embed_dim)
		self.audio_encoder = nn.GRU(
								input_size=self.embed_dim,
								hidden_size=self.embed_dim,
								num_layers=self.num_layers,
								dropout=opts.dropout,
								batch_first=True,
								bidirectional=True
							)

		self.text_proj = nn.Linear(self.text_dim, self.embed_dim)
		self.text_encoder = nn.GRU(
	  							input_size=self.embed_dim,
			 					hidden_size=self.embed_dim,
				  				num_layers=self.num_layers,
					  			dropout=opts.dropout,
								batch_first=True,
								bidirectional=True
			  				)

		self.classifier = nn.Sequential(
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))

	def forward(self, ori_images, gh_feats, cur_context_gh_feats, audio_feats, text_feats):
		batch_size = ori_images.shape[0]
		num_frames = cur_context_gh_feats.shape[1]
		ori_visual_reps = self.vision_encoder_1(ori_images).reshape(batch_size, -1)
		ori_visual_reps = self.static_proj_1(ori_visual_reps)
		# super_res_visual_reps = self.vision_encoder_2(super_res_images).reshape(batch_size, -1)
		# super_res_visual_reps = self.static_proj_2(super_res_visual_reps)
  
		gh_feats = self.static_proj_3(gh_feats.reshape(batch_size, -1))

		context_gh_feats = self.gh_feat_proj(cur_context_gh_feats.reshape(batch_size*num_frames, -1))
		context_gh_feats = context_gh_feats.reshape(batch_size, num_frames, -1)
		gh_feat_outputs, _ = self.gh_feat_encoder(context_gh_feats) # BxTx128
		gh_reps = gh_feat_outputs[:,-1,:]

		audio_feats = self.audio_proj(audio_feats)
		audio_outputs, _ = self.audio_encoder(audio_feats) # BxTx128
		audio_reps = audio_outputs[:,-1,:]

		text_feats = self.text_proj(text_feats)
		text_outputs, _ = self.text_encoder(text_feats) # BxTx128
		text_reps = text_outputs[:,-1,:]

		labels = self.classifier(torch.cat([ori_visual_reps, gh_feats, gh_reps, audio_reps, text_reps], dim=1))

		return labels

class SwinTransformerTiny(nn.Module):
	def __init__(self, opts):
		super(SwinTransformerTiny, self).__init__()

		self.encoder = swin_transformer_tiny()
		self.classifier = nn.Sequential(
			nn.Linear(49*768, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, 8))

	def forward_prime(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels = self.classifier(features)

		return features, labels

	def forward(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels = self.classifier(features)

		return labels
