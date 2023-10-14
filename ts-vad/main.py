import argparse, glob, os, warnings, time
from tools.tools import *
from trainer import *
from dataLoader import *

parser = argparse.ArgumentParser(description = "Target Speaker VAD")

### Training setting
# parser.add_argument('--max_epoch',  type=int,   default=100,      help='Maximum number of epochs')
parser.add_argument('--max_epoch',  type=int,   default=10,      help='Maximum number of epochs')
# parser.add_argument('--warm_up_epoch',  type=int, default=10,      help='Maximum number of epochs')
parser.add_argument('--warm_up_epoch',  type=int, default=5,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=10,      help='Batch size')
parser.add_argument('--rs_len',     type=float, default=16,      help='Input length (second) of reference speech')
# parser.add_argument('--n_cpu',      type=int,   default=12,       help='Number of loader threads')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,        help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.0001,    help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.90,     help='Learning rate decay every [test_step] epochs')

### Testing setting
parser.add_argument('--test_shift', type=float, default=16,      help='Input shift (second) for testing')
parser.add_argument('--min_silence', type=float, default=0.32,      help='Remove the speech with short slience during testing')
parser.add_argument('--min_speech', type=float, default=0.00,      help='Combine the short speech during testing')
parser.add_argument('--threshold', type=float, default=0.50,      help='The threshold during testing')
parser.add_argument('--init_model',  type=str,   default="/home/users/ntu/tlkushag/scratch/TSVAD_pytorch/ts-vad/pretrained_models/ts-vad.model",  help='Init TS-VAD model from pretrain')

### Data path
parser.add_argument('--train_list', type=str,   default="/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_dev/data/ts_dev.json",     help='The path of the training list')
parser.add_argument('--train_path', type=str,   default="/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_dev/data", help='The path of the training data')
parser.add_argument('--eval_list',  type=str,   default="/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data/ts_eval.json",      help='The path of the evaluation list')
parser.add_argument('--eval_path',  type=str,   default="/home/users/ntu/tlkushag/scratch/data08/dihard/third_dihard_challenge_eval/data", help='The path of the evaluation data')
parser.add_argument('--save_path',  type=str,    default="", help='Path to save the clean list')
parser.add_argument('--musan_path',  type=str,   default="/home/users/ntu/tlkushag/scratch/data08/Others/musan_split", help='The path of the evaluation data')
parser.add_argument('--rir_path',  type=str,   default="/home/users/ntu/tlkushag/scratch/data08/Others/RIRS_NOISES/simulated_rirs", help='The path of the evaluation data')

### Others
parser.add_argument('--speech_encoder_pretrain',  type=str,   default="/home/users/ntu/tlkushag/scratch/TSVAD_pytorch/ts-vad/pretrained_models/WavLM-Base+.pt",  help='Path of the pretrained speech_encoder')
parser.add_argument('--train',   dest='train', action='store_true', help='Do training')
parser.add_argument('--eval',    dest='eval', action='store_true', help='Do evaluation')

## Init folders, trainer and loader
args = init_system(parser.parse_args())
print(f"[System initialization completed]")

s = init_trainer(args)
print(f"[Trainer initialization completed]")

args = init_loader(args)
print(f"[Loader initialization completed]")

# print args
print("\n### Parameters ###")
for arg in vars(args):
	print(arg, getattr(args, arg))
print("### Parameters ###\n")

args.eval=True
## Evaluate only
if args.eval == True:
	s.eval_network(args)
	quit()

## Training
if args.train == True:
	while args.epoch < args.max_epoch:
		for param in s.ts_vad.speech_encoder.parameters():
			if args.epoch < args.warm_up_epoch:
				param.requires_grad = False
			else:
				param.requires_grad = True
		args = init_loader(args) # Random the training list for more samples
		s.train_network(args)
		if args.epoch % args.test_step == 0:
			s.save_parameters(args.model_save_path + "/model_%04d.model"%args.epoch)
			s.eval_network(args)
		args.epoch += 1
	quit()