import torch, sys, os, tqdm, numpy, soundfile, time, pickle, glob, random, scipy, signal, subprocess
import torch.nn as nn
from tools.tools import *
from loss import *
from model.ts_vad import TS_VAD
# from model.ts_vad_light import TS_VAD
from collections import defaultdict, OrderedDict
from torch.cuda.amp import autocast,GradScaler
from scipy import signal

def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	if args.init_model != "":
		print("Model %s loaded from pretrain!"%args.init_model)
		s.load_parameters(args.init_model)		
	elif len(args.modelfiles) >= 1:
		print("Model %s loaded from previous state!"%args.modelfiles[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles[-1])
	return s

class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		# TODO: Change back to cuda
		self.ts_vad          = TS_VAD(args).cuda()
		# self.ts_vad          = TS_VAD(args)
		# TODO: Change back to cuda
		self.ts_loss         = Loss().cuda()	
		# self.ts_loss         = Loss()
		self.optim           = torch.optim.AdamW(self.parameters(), lr = args.lr)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)
		# print("Model para number = %.2f"%(sum(param.numel() for param in self.ts_vad.parameters()) / 1e6))

	def train_network(self, args):
		self.train()
		scaler = GradScaler()
		self.scheduler.step(args.epoch - 1)
		index, nloss = 0, 0
		lr = self.optim.param_groups[0]['lr']
		time_start = time.time()

		for num, (rs, ts, labels) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()
			# TODO: Change back to cuda
			labels  = torch.tensor(labels, dtype=torch.float32).cuda()	
			# labels  = torch.tensor(labels, dtype=torch.float32)	
			with autocast():
				# TODO: Change back to cuda
				rs_embeds  = self.ts_vad.rs_forward(rs.cuda())
				# rs_embeds  = self.ts_vad.rs_forward(rs)
				# TODO: Change back to cuda
				ts_embeds  = self.ts_vad.ts_forward(ts.cuda())
				# ts_embeds  = self.ts_vad.ts_forward(ts)
				outs       = self.ts_vad.cat_forward(rs_embeds, ts_embeds)
				loss, _    = self.ts_loss.forward(outs, labels)	
			scaler.scale(loss).backward()
			scaler.step(self.optim)
			scaler.update()
			index += len(labels)
			nloss += loss.detach().cpu().numpy()
			time_used = time.time() - time_start
			sys.stderr.write("Train: [%2d] %.2f%% (est %.1f mins) Lr: %6f, Loss: %.5f\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, \
			lr, nloss/(num)))
			sys.stderr.flush()			
		sys.stdout.write("\n")

		args.score_file.write("Train: %d epoch, LR %f, LOSS %f\n"%(args.epoch, lr, nloss/num))
		args.score_file.flush()
		return
		
	def eval_network(self, args):
		self.eval()
		index, nloss = 0, 0
		time_start = time.time()
		res_dict = defaultdict(lambda: defaultdict(list))
		rttm = open(args.rttm_save_path, "w")		
		for num, (rs, ts, labels, filename, speaker_id, start) in enumerate(args.evalLoader, start = 1):
			# TODO: Change back to cuda
			labels  = torch.tensor(labels, dtype=torch.float32).cuda()
			# labels  = torch.tensor(labels, dtype=torch.float32)
			with torch.no_grad():		
				# TODO: Change back to cuda
				rs_embeds  = self.ts_vad.rs_forward(rs.cuda())
				# rs_embeds  = self.ts_vad.rs_forward(rs)
				# print(f"rs_embeding extracted")
				
				# TODO: Change back to cuda
				ts_embeds  = self.ts_vad.ts_forward(ts.cuda())
				# ts_embeds  = self.ts_vad.ts_forward(ts)
				# print(f"ts_embeding extracted")

				outs       = self.ts_vad.cat_forward(rs_embeds, ts_embeds)
				# print("outs calculated")
				loss, outs   = self.ts_loss.forward(outs, labels)
				# print("loss calculated")

				B, _, T = outs.shape
				labels = labels.cpu().numpy()
				for b in range(B):
					for t in range(T):
						n = max(speaker_id[b,:].cpu().numpy())						
						for i in range(n):
							id = speaker_id[b,i].cpu().numpy()
							name = filename[b]
							out = outs[b,i,t]
							t0 = start[b].numpy()
							res_dict[str(name) + '-' + str(id)][t0 + t].append(out)
				# print("res_dict updated")
			index += len(labels)
			nloss += loss.detach().cpu().numpy()
			time_used = time.time() - time_start
			sys.stderr.write("Eval: [%2d] %.2f%% (est %.1f mins) Loss: %.5f\r"%\
			(args.epoch, 100 * (num / args.evalLoader.__len__()), time_used * args.evalLoader.__len__() / num / 60, \
			nloss/(num)))
			sys.stderr.flush()		
		
		for filename in tqdm.tqdm(res_dict):
			name, speaker_id =filename.split('-')
			labels = res_dict[filename]
			ave_labels = []
			for key in labels:	
				ave_labels.append(numpy.mean(labels[key]))
			labels = signal.medfilt(ave_labels, 21)			
			labels = change_zeros_to_ones(labels, args.min_silence, args.threshold)
			labels = change_ones_to_zeros(labels, args.min_speech, args.threshold)
			start, duration = 0, 0
			for i, label in enumerate(labels):
				if label == 1:
					duration += 0.04
				else:
					if duration != 0:
						line = "SPEAKER " + str(name) + ' 1 %.3f'%(start) + ' %.3f ' %(duration) + '<NA> <NA> ' + str(speaker_id) + ' <NA> <NA>\n'
						rttm.write(line)
						duration = 0
					start = i * 0.04
			if duration != 0:
				line = "SPEAKER " + str(name) + ' 1 %.3f'%(start) + ' %.3f ' %(duration) + '<NA> <NA> ' + str(speaker_id) + ' <NA> <NA>\n'
				rttm.write(line)
		rttm.close()
		print('\n')
		print("For collar value 0.0:")
		out = subprocess.check_output(['perl', '/home/users/ntu/tlkushag/scratch/TSVAD_pytorch/ts-vad/tools/SCTK-2.4.12/src/md-eval/md-eval.pl', '-c 0.0', '-s %s'%(args.rttm_save_path), '-r /home/users/ntu/tlkushag/scratch/TSVAD_pytorch/wespeaker_alimeeting/exp/label/all.rttm'])
		out = out.decode('utf-8')
		DER, MS, FA, SC = float(out.split('/')[0]), float(out.split('/')[1]), float(out.split('/')[2]), float(out.split('/')[3])
		print("DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%\n"%(DER, MS, FA, SC))

		print("For collar value 0.25:")
		out = subprocess.check_output(['perl', '/home/users/ntu/tlkushag/scratch/TSVAD_pytorch/ts-vad/tools/SCTK-2.4.12/src/md-eval/md-eval.pl', '-c 0.25', '-s %s'%(args.rttm_save_path), '-r /home/users/ntu/tlkushag/scratch/TSVAD_pytorch/wespeaker_alimeeting/exp/label/all.rttm'])
		out = out.decode('utf-8')
		DER, MS, FA, SC = float(out.split('/')[0]), float(out.split('/')[1]), float(out.split('/')[2]), float(out.split('/')[3])
		print("DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%\n"%(DER, MS, FA, SC))
		
		args.score_file.write("Eval: %d epoch, DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%, LOSS %f\n"%(args.epoch, DER, MS, FA, SC, nloss/num))
		args.score_file.flush()

	def save_parameters(self, path):
		model = OrderedDict(list(self.ts_vad.state_dict().items()) + list(self.ts_loss.state_dict().items()))
		torch.save(model, path)

	def load_parameters(self, path):
		selfState = self.state_dict()
		# TODO: Change back to cuda
		loadedState = torch.load(path)
		# loadedState = torch.load(path, map_location="cpu")
		for name, param in loadedState.items():
			origName = name
			if name not in selfState:
				name = 'ts_vad.' + name
				if name not in selfState:
					name = name.replace('ts_vad', 'ts_loss')
					if name not in selfState:
						print("%s is not in the model."%origName)
						continue
			if selfState[name].size() != loadedState[origName].size():
				sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
				continue
			selfState[name].copy_(param)