import glob, tqdm, os, soundfile, copy, json, argparse, numpy, torch, wave, heapq
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
	"""
	Remove overlapping intervals between two lists of intervals.

	Args:
		aa (list): The first list of intervals.
		bb (list): The second list of intervals.

	Returns:
		list: A new list of intervals with overlapping intervals removed.
	"""
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
	"""
	Initializes a speaker encoder model with pre-trained weights from a given source file.

	Args:
		source (str): The path to the source file containing the pre-trained weights.

	Returns:
		speaker_encoder (ECAPA_TDNN): The initialized speaker encoder model with pre-trained weights.
	"""
	speaker_encoder = ECAPA_TDNN(C=1024).cuda()
	speaker_encoder.eval()
	loadedState = torch.load(source, map_location="cuda")
	selfState = speaker_encoder.state_dict()
	for name, param in loadedState.items():
		if name in selfState:
			selfState[name].copy_(param)
	for param in speaker_encoder.parameters():
		param.requires_grad = False 
	return speaker_encoder

def extract_embeddings(batch, model):
	"""
	Extracts embeddings for a batch of data using a given model.

	Args:
		batch (list): A list of data samples.
		model: A PyTorch model used for extracting embeddings.

	Returns:
		embeddings: A tensor of embeddings for the input batch.
	"""
	batch = torch.stack(batch)
	with torch.no_grad():
		embeddings = model.forward(batch.cuda())
	return embeddings

def get_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data_path', help='the path for dihard3')
	parser.add_argument('--type', help='dev or eval')
	parser.add_argument('--source', help='the part for the speaker encoder')
	parser.add_argument('--length_embedding', type=float, default=6, help='length of embeddings, seconds')
	parser.add_argument('--step_embedding', type=float, default=1, help='step of embeddings, seconds')
	parser.add_argument('--batch_size', type=int, default=96, help='step of embeddings, seconds')

	args = parser.parse_args()
	args.path = os.path.join(args.data_path, 'third_dihard_challenge_%s/data'%(args.type))
	# args.path_flac = os.path.join(args.path, 'flac')
	args.path_wav = os.path.join(args.path, 'wav')
	args.path_rttm = os.path.join(args.path, 'rttm')
	# args.target_flac = os.path.join(args.path, 'target_audio')
	args.target_wav = os.path.join(args.path, 'target_audio')
	args.target_embedding = os.path.join(args.path, 'target_embedding')
	args.out_text = os.path.join(args.path, 'ts_%s.json'%(args.type))
	return args

def save_target_audio(target_audio_path, segments, flac_path):
	# each segment is a Segment object with uttid, spkr, stime, etime, text, name

	# read the original audio
	audio, sr = soundfile.read(flac_path)

	# save the original audio
	soundfile.write(os.path.join(target_audio_path, 'all.flac'), audio, sr)

	# save the audio for each speaker using the segments. Segments have been sorted by stime. For each spkr, we combine all the segments and save as one flac file.

	# get the unique speakers
	speakers = set([seg.spkr for seg in segments])

	# for each speaker, combine the segments and save as one flac file
	for spkr in speakers:
		# get the segments for this speaker
		spkr_segments = [seg for seg in segments if seg.spkr == spkr]
		# combine the segments
		spkr_audio = numpy.concatenate([audio[int(seg.stime*sr):int(seg.etime*sr)] for seg in spkr_segments])
		# save the audio
		soundfile.write(os.path.join(target_audio_path, spkr + '.flac'), spkr_audio, sr)


def main():
	args = get_args()
	
	outs = open(args.out_text, "w")
	rttms = glob.glob(args.path_rttm + '/*')

	for rttm in tqdm.tqdm(rttms):
		# convert rttm file to a json file.
		# The format of the rttm file is: SPEAKER <filename> 1 <turn_onset> <turn_duration> <NA> <NA> <speaker_id> <NA> <NA>
		# The format of the json file is: [ {"filename":"filename", "speaker_key":1, "speaker_id":"speakerid", "labels": [0,1,0]}]
		segments = []
		uttid2spkr = {}
		uttid2name = {}

		with open(rttm, 'r') as f:
			for line in f:
				line = line.strip().split()
				uttid = line[1]
				stime = float(line[3])
				etime = float(line[3]) + float(line[4])
				name = line[7]
				# spkr is line[7] without the prefix "speaker"
				spkr = line[7][7:]
				segments.append(Segment(uttid, spkr, stime, etime, '', name))
				uttid2spkr[uttid] = spkr
				uttid2name[uttid] = name

		segments = sorted(segments, key=lambda x: x.spkr)

		intervals = defaultdict(list)
		new_intervals = defaultdict(list)
		
		dic = defaultdict()
		# Summary the intervals for all speakers
		for i in range(len(segments)):
			interval = [segments[i].stime, segments[i].etime]
			intervals[segments[i].spkr].append(interval)
			dic[str(segments[i].uttid) + '_' + str(segments[i].spkr)] = segments[i].name.split('_')[-1]

		# Remove the overlapped speeech
		for key in intervals:
			new_interval = intervals[key]
			for o_key in intervals:
				if o_key != key:
					new_interval = remove_overlap(copy.deepcopy(new_interval), copy.deepcopy(intervals[o_key]))
			new_intervals[key] = new_interval
		
		wav_file = glob.glob(os.path.join(args.path_wav, uttid) + '*.wav')[0]
		orig_audio, _ = soundfile.read(wav_file)
		length = len(orig_audio) 

		# # Cut and save the clean speech part
		id_full = wav_file.split('/')[-1][:-4]
		
		# create a dictionary to store new_audio for each speaker and it's output_wav
		new_audio_dict = {}

		for key in new_intervals:
			output_dir = os.path.join(args.target_wav, id_full)
			os.makedirs(output_dir, exist_ok = True)
			output_wav = os.path.join(output_dir, str(key) + '.wav')
			new_audio = []
			labels = [0] * int(length / 16000 * 25) # 40ms, one label        
			for interval in new_intervals[key]:
				s, e = interval
				for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
					labels[i] = 1 
				s *= 16000
				e *= 16000
				new_audio.extend(orig_audio[int(s):int(e)])
			
			new_audio_dict[key] = [new_audio, output_wav]

		# filter new_audio_dict to get the longest 4 speakers
		new_audio_dict = dict(heapq.nlargest(4, new_audio_dict.items(), key=lambda item: len(item[1][0])))
		mapping = {}
		mapping1 = {}

		# rename the output_wav in new_audio_dict to start from 1.wav
		for i, key in enumerate(new_audio_dict):
			# get filename without extension from new_audio_dict[key][1]
			filename = new_audio_dict[key][1].split('/')[-1].split('.')[0]
			# utterance_id is the 'rttm' filename without the .rttm extension
			utterance_id = rttm.split('/')[-1].split('.')[0]

			# add mapping, with key as filename, utterance_id and value as str(i + 1)
			mapping[utterance_id + '_' + str(i + 1)] = filename
			mapping1[filename] = str(i + 1)

			new_audio_dict[key][1] = os.path.join(output_dir, str(i + 1) + '.wav')

		# initialise good_speakers set
		good_speakers = set()

		# use soundfile to write the audios in new_audio_dict to output_wav
		for key in new_audio_dict:
			if (new_audio_dict[key][0] != []):
				if (len(new_audio_dict[key][0]) < 16000 * 6):
					# get the speaker_id from new_audio_dict[key][1]
					speaker_idd = new_audio_dict[key][1].split('/')[-1].split('.')[0]
				else:
					soundfile.write(new_audio_dict[key][1], new_audio_dict[key][0], 16000)
					speaker_idd = new_audio_dict[key][1].split('/')[-1].split('.')[0]
					good_speakers.add(speaker_idd)
			
		output_wav = os.path.join(output_dir, 'all.wav')
		soundfile.write(output_wav, orig_audio, 16000)

		# Save the labels
		for key in new_intervals:
			speaker_id = "speaker" + str(key)
			if (str(key) not in mapping.values()):
				continue

			labels = [0] * int(length / 16000 * 25) # 40ms, one label        
			for interval in intervals[key]:
				s, e = interval
				for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
					labels[i] = 1

			if mapping1[str(key)] in good_speakers:
				res = {'filename':id_full, 'speaker_key':mapping1[key], 'speaker_id': speaker_id, 'labels':labels}
				json.dump(res, outs)

				outs.write('\n')

	# Extract embeddings
	files = sorted(glob.glob(args.target_wav + "/*/*.wav"))
	model = init_speaker_encoder(args.source)

	for file in tqdm.tqdm(files):
		if 'all.wa' not in file:
			batch = []
			embeddings = []

			wav_length = wave.open(file, 'rb').getnframes() # entire length for target speech

			for start in range(0, wav_length - int(args.length_embedding * 16000), int(args.step_embedding * 16000)):
				stop = start + int(args.length_embedding * 16000)
				target_speech, _ = soundfile.read(file, start = start, stop = stop)
				target_speech = torch.FloatTensor(numpy.array(target_speech))
				batch.append(target_speech)
				if len(batch) == args.batch_size:                
					embeddings.extend(extract_embeddings(batch, model))
					batch = []
			if len(batch) != 0:
				embeddings.extend(extract_embeddings(batch, model))             
			if embeddings:
				embeddings = torch.stack(embeddings)
				output_file = args.target_embedding + '/' + file.split('/')[-2] + '/' + file.split('/')[-1].replace('.wav', '.pt')
				os.makedirs(os.path.dirname(output_file), exist_ok = True)
				torch.save(embeddings, output_file)
			else:
				print(file)


if __name__ == '__main__':
	main()