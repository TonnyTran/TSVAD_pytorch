import glob, tqdm, os, textgrid, soundfile, copy, json, argparse, numpy, torch, wave
from collections import defaultdict
from speaker_encoder import ECAPA_TDNN

class Segment(object):
	def __init__(self, uttid, spkr, stime, etime, text, name):
		self.uttid = uttid
		self.spkr = spkr
		self.stime = round(stime, 2)
		self.etime = round(etime, 2)
		self.text = text
		self.name = name

	def change_stime(self, time):
		self.stime = time

	def change_etime(self, time):
		self.etime = time

def remove_overlap(aa, bb):
	# Sort the intervals in both lists based on their start time
	a = aa.copy()
	b = bb.copy()
	a.sort()
	b.sort()

	# Initialize the new list of intervals
	result = []

	# Initialize variables to keep track of the current interval in list a and the remaining intervals in list b
	i = 0
	j = 0

	# Iterate through the intervals in list a
	while i < len(a):
		# If there are no more intervals in list b or the current interval in list a does not overlap with the current interval in list b, add it to the result and move on to the next interval in list a
		if j == len(b) or a[i][1] <= b[j][0]:
			result.append(a[i])
			i += 1
		# If the current interval in list a completely overlaps with the current interval in list b, skip it and move on to the next interval in list a
		elif a[i][0] >= b[j][0] and a[i][1] <= b[j][1]:
			i += 1
		# If the current interval in list a partially overlaps with the current interval in list b, add the non-overlapping part to the result and move on to the next interval in list a
		elif a[i][0] < b[j][1] and a[i][1] > b[j][0]:
			if a[i][0] < b[j][0]:
				result.append([a[i][0], b[j][0]])
			a[i][0] = b[j][1]
		# If the current interval in list a starts after the current interval in list b, move on to the next interval in list b
		elif a[i][0] >= b[j][1]:
			j += 1

	# Return the new list of intervals
	return result


def init_speaker_encoder(source):
	# TODO: change to cuda
	speaker_encoder = ECAPA_TDNN(C=1024).cuda()
	# speaker_encoder = ECAPA_TDNN(C=1024)
	speaker_encoder.eval()
	# TODO: change to cuda
	loadedState = torch.load(source, map_location="cuda")
	# loadedState = torch.load(source, map_location="cpu")
	selfState = speaker_encoder.state_dict()
	for name, param in loadedState.items():
		if name in selfState:
			selfState[name].copy_(param)
	for param in speaker_encoder.parameters():
		param.requires_grad = False 
	return speaker_encoder

def extract_embeddings(batch, model):	
	batch = torch.stack(batch)    
	with torch.no_grad():
		# TODO: change to cuda
		embeddings = model.forward(batch.cuda())
		# embeddings = model.forward(batch)
	return embeddings

def get_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data_path', default='/home/users/ntu/tlkushag/scratch/data08/dihard', help='the path for the dihard')
	parser.add_argument('--type', default='dev', help='dev or eval')
	parser.add_argument('--source', default='/home/users/ntu/tlkushag/scratch/TSVAD_pytorch/wespeaker_alimeeting/pretrained_models/ecapa-tdnn.model', help='the part for the speaker encoder')
	parser.add_argument('--length_embedding', type=float, default=6, help='length of embeddings, seconds')
	parser.add_argument('--step_embedding', type=float, default=1, help='step of embeddings, seconds')
	parser.add_argument('--batch_size', type=int, default=96, help='step of embeddings, seconds')

	args = parser.parse_args()
	args.path = os.path.join(args.data_path, 'third_dihard_challenge_%s'%(args.type),'data')
	args.path_wav = os.path.join(args.path, 'wav')
	args.path_grid = os.path.join(args.path, 'textgrid')
	args.target_wav = os.path.join(args.path, 'target_audio')
	args.target_embedding = os.path.join(args.path, 'target_embedding')
	args.out_text = os.path.join(args.path, 'ts_%s.json'%(args.type))
	return args

def main():
	args = get_args()
	print('Arguments:')
	for arg in vars(args):
		print(arg + ':', getattr(args, arg))

	# text_grids = glob.glob(args.path_grid + '/*')
	# outs = open(args.out_text, "w") #ts-dev.json

	# print("Processing textgrids")
	# for text_grid in tqdm.tqdm(text_grids):
	# 	tg = textgrid.TextGrid.fromFile(text_grid)
	# 	segments = []
	# 	spk = {}
	# 	num_spk = 1
	# 	uttid = text_grid.split('/')[-1][:-9]
	# 	for i in range(tg.__len__()):
	# 		for j in range(tg[i].__len__()):
	# 			if tg[i][j].mark:
	# 				if tg[i].name not in spk:
	# 					spk[tg[i].name] = num_spk
	# 					num_spk += 1
	# 				segments.append(Segment(
	# 						uttid,
	# 						spk[tg[i].name],
	# 						tg[i][j].minTime,
	# 						tg[i][j].maxTime,
	# 						tg[i][j].mark.strip(),
	# 						tg[i].name
	# 					)
	# 				)
	# 	segments = sorted(segments, key=lambda x: x.spkr)
		
	# 	intervals = defaultdict(list)
	# 	new_intervals = defaultdict(list)

	# 	dic = defaultdict()
	# 	# Summary the intervals for all speakers
	# 	for i in range(len(segments)):
	# 		interval = [segments[i].stime, segments[i].etime]
	# 		intervals[segments[i].spkr].append(interval)
	# 		dic[str(segments[i].uttid) + '_' + str(segments[i].spkr)] = segments[i].name.split('_')[-1]

	# 	# Remove the overlapped speeech    
	# 	for key in intervals:
	# 		new_interval = intervals[key]
	# 		for o_key in intervals:
	# 			if o_key != key:                
	# 				new_interval = remove_overlap(copy.deepcopy(new_interval), copy.deepcopy(intervals[o_key]))
	# 		new_intervals[key] = new_interval

	# 	wav_file = glob.glob(os.path.join(args.path_wav, uttid) + '*.wav')[0]
	# 	orig_audio, _ = soundfile.read(wav_file,always_2d=True)
	# 	orig_audio = orig_audio[:,0]
	# 	length = len(orig_audio) 

	# 	# # Cut and save the clean speech part
	# 	id_full = wav_file.split('/')[-1][:-4]
	# 	if(args.type == 'dev'):
	# 		room_id = id_full[:11]
	# 	else:
	# 		room_id = id_full[:12]
	# 	for key in new_intervals:
	# 		output_dir = os.path.join(args.target_wav, id_full)
	# 		os.makedirs(output_dir, exist_ok = True)
	# 		output_wav = os.path.join(output_dir, str(key) + '.wav')
	# 		new_audio = []
	# 		labels = [0] * int(length / 16000 * 25) # 40ms, one label        
	# 		for interval in new_intervals[key]:
	# 			s, e = interval
	# 			for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
	# 				labels[i] = 1 
	# 			s *= 16000
	# 			e *= 16000
	# 			new_audio.extend(orig_audio[int(s):int(e)])
	# 		soundfile.write(output_wav, new_audio, 16000)
	# 	output_wav = os.path.join(output_dir, 'all.wav')
	# 	soundfile.write(output_wav, orig_audio, 16000)

	# 	# Save the labels
	# 	for key in intervals:
	# 		labels = [0] * int(length / 16000 * 25) # 40ms, one label        
	# 		for interval in intervals[key]:
	# 			s, e = interval
	# 			for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
	# 				labels[i] = 1

	# 		room_speaker_id = room_id + '_' + str(key)
	# 		if(room_speaker_id not in dic):
	# 			print("Error: ", room_speaker_id)
	# 			continue
	# 		speaker_id = dic[room_speaker_id]

	# 		res = {'filename':id_full, 'speaker_key':key, 'speaker_id': speaker_id, 'labels':labels}
	# 		json.dump(res, outs)
	# 		outs.write('\n')

	# Extract embeddings
	print("Extracting embeddings")
	files_with_zero_length = 0

	files = sorted(glob.glob(args.target_wav + "/*/*.wav"))
	model = init_speaker_encoder(args.source)

	for file in tqdm.tqdm(files):
		if 'all.wav' not in file:
			batch = []
			embeddings = []
			wav_length = wave.open(file, 'rb').getnframes() # entire length for target speech in frames which is seconds * 16000 * sampling rate				
			print("File: ",file)
			print("Wav length: ", wav_length)
			if (wav_length - int(args.length_embedding * 16000)) <= 0:
				# set embedding to torch.Size([96, 192])
				files_with_zero_length += 1
				embedding = torch.zeros((96, 192))
				embeddings.extend(embedding)
				embeddings = torch.stack(embeddings)
				output_file = args.target_embedding + '/' + file.split('/')[-2] + '/' + file.split('/')[-1].replace('.wav', '.pt')
				os.makedirs(os.path.dirname(output_file), exist_ok = True)
				torch.save(embeddings, output_file)
				continue

			for start in range(0, wav_length - int(args.length_embedding * 16000), int(args.step_embedding * 16000)):
				stop = start + int(args.length_embedding * 16000)
				target_speech, _ = soundfile.read(file, start = start, stop = stop)
				target_speech = torch.FloatTensor(numpy.array(target_speech))
				batch.append(target_speech)
				if len(batch) == args.batch_size:                
					embedding = extract_embeddings(batch, model)
					embeddings.extend(embedding)					
					batch = []
			
			if len(batch) != 0:
				embeddings.extend(extract_embeddings(batch, model))   

			embeddings = torch.stack(embeddings)
			output_file = args.target_embedding + '/' + file.split('/')[-2] + '/' + file.split('/')[-1].replace('.wav', '.pt')
			os.makedirs(os.path.dirname(output_file), exist_ok = True)
			torch.save(embeddings, output_file)

	print("Number of files with zero length: ", files_with_zero_length)

if __name__ == '__main__':
	main()