import argparse, glob, os, torch, warnings, time
from tools import *
from trainer import *
from dataLoader import *

parser = argparse.ArgumentParser(description = "Target Speaker VAD")

### Training setting
parser.add_argument('--max_epoch',  type=int,   default=100,      help='Maximum number of epochs')
parser.add_argument('--warm_up_epoch',  type=int,   default=30,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=30,      help='Batch size')
parser.add_argument('--ts_len',  type=int,   default=6000,      help='Input ms of target speaker utterance')
parser.add_argument('--rs_len',  type=int,   default=6000,      help='Input ms of reference speech')
parser.add_argument('--n_cpu',      type=int,   default=12,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,        help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.0001,    help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.99,     help='Learning rate decay every [test_step] epochs')

### Data path
parser.add_argument('--train_list', type=str,   default="/data08/alimeeting/ts_Train.json",     help='The path of the training list')
parser.add_argument('--train_path', type=str,   default="/data08/alimeeting/Train_Ali_far/target_audio_dir", help='The path of the training data')
parser.add_argument('--eval_list',  type=str,   default="/data08/alimeeting/ts_Eval.json",      help='The path of the evaluation list')
parser.add_argument('--eval_path',  type=str,   default="/data08/alimeeting/Eval_Ali_far/target_audio_dir", help='The path of the evaluation data')
parser.add_argument('--save_path',  type=str,    default="", help='Path to save the clean list')
parser.add_argument('--musan_path',   type=str, default="/data08/Others/musan_split")
parser.add_argument('--rir_path',     type=str, default="/data08/Others/RIRS_NOISES/simulated_rirs")

### Initial modal path
parser.add_argument('--speaker_encoder',  type=str,   default="pretrain/resnet.pt",  help='Path of the pretrained speaker_encoder')
parser.add_argument('--ref_speech_encoder',  type=str,   default="pretrain/WavLM-Base+.pt",  help='Path of the pretrained speech_encoder')

###  Others
parser.add_argument('--train',   dest='train', action='store_true', help='Do training')
parser.add_argument('--eval',    dest='eval', action='store_true', help='Do evaluation')

## Init folders
args = init_system(parser.parse_args())
## Init trainer
s = init_trainer(args)
## Init loader
args = init_loader(args)

## Evaluate only
if args.eval == True:
	s.eval_network(args)
	quit()

## Train only
if args.train == True:
	while args.epoch < args.max_epoch:
		for param in s.ts_vad.speech_encoder.parameters():
			if args.epoch < args.warm_up_epoch:
				param.requires_grad = False
			else:
				param.requires_grad = True
		args = init_loader(args)
		s.train_network(args)
		if args.epoch % args.test_step == 0:
			s.save_parameters(args.model_save_path + "/model_%04d.model"%args.epoch)
			s.eval_network(args)
		args.epoch += 1
	quit()