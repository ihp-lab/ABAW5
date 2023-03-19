import argparse

from solver_multimodal import solver_multimodal
from utils import set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)

# storage
parser.add_argument('--data_root', type=str, default='../data/image/')
parser.add_argument('--ckpt_path', type=str, default='./checkpoints')

# multimodal
parser.add_argument('--audio_feat_path', type=str, default='../data/audio_hubert_xlarge_tapt_feats/')
parser.add_argument('--text_feat_path', type=str, default='../data/roberta_feats_all.pkl')
parser.add_argument('--text_timestamp_path', type=str, default='../data/ABAW_transcript_word_timestamp/')
parser.add_argument('--video_fps_info_path', type=str, default='../data/video_fps_info.csv')

# data
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--num_labels', type=int, default=12)
parser.add_argument('--num_frames', type=int, default=8)

# model
parser.add_argument('--pretrain', type=str, default='none')
parser.add_argument('--model', type=str, default='v2')
parser.add_argument('--model_name', type=str, default='none')

# training
parser.add_argument('--num_epochs', type=int, default=15)
parser.add_argument('--interval', type=int, default=500)
parser.add_argument('--threshold', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--loss', type=str, default='weighted')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--clip', type=int, default=1.0)
parser.add_argument('--when', type=int, default=5, help='when to decay learning rate')
parser.add_argument('--patience', type=int, default=5, help='early stopping')

opts = parser.parse_args()
print(opts)

# Fix random seed
set_seed(opts.seed)

# Setup solver
solver = solver_multimodal(opts).cuda()

# Start training
solver.run()
