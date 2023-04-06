import glob, numpy, os, random, soundfile, torch, json
from collections import defaultdict

def init_loader(args):
	trainloader = train_loader(**vars(args))
	args.trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
	evalLoader = eval_loader(**vars(args))
	args.evalLoader = torch.utils.data.DataLoader(evalLoader, batch_size = args.batch_size, shuffle = False, num_workers = args.n_cpu, drop_last = False)
	return args

class train_loader(object):
	def __init__(self, train_list, train_path, rs_len, **kwargs):
		self.train_path = train_path	
		self.rs_len = int(rs_len * 25) # Number of frames for reference speech

		self.data_list = []
		self.label_dic = defaultdict(list)

		lines = open(train_list).read().splitlines()
		filename_set = set()
		# Load the data and labels
		for line in lines:
			dict = json.loads(line)
			length = len(dict['labels']) # Number of frames (1s = 25 frames)
			filename = dict['filename']
			labels = dict['labels']
			speaker_id = str(dict['speaker_key'])
			full_id = filename + '_' + speaker_id
			self.label_dic[full_id] = labels
			if filename in filename_set:
				pass
			else:
				filename_set.add(filename)
				start_place = random.randint(0, rs_len) * 25 # Random start time, to make more training samples
				for start in range(start_place, length - self.rs_len, self.rs_len):
					folder = self.train_path + '/target_audio_dir/' + filename + '/*.wav'
					audios = glob.glob(folder)
					num_speaker = len(audios) - 1 # The total number of speakers, 2 or 3 or 4
					data_intro = [filename, num_speaker, start, start + self.rs_len]
					self.data_list.append(data_intro)

	def __getitem__(self, index):
		# T: number of frames (1s contrains 25 frames)
		# ref_speech : 16000 * T
		# labels : 4, T
		# target_speech: 4, 192
		file, num_speaker, start, stop = self.data_list[index]
		speaker_ids = self.get_ids(num_speaker)
		ref_speech, labels = self.load_rs(file, speaker_ids, start, stop)
		target_speech = self.load_ts(file, speaker_ids)
		return ref_speech, target_speech, labels
	
	def get_ids(self, num_speaker):
		speaker_ids = []
		if num_speaker == 2:
			speaker_ids = [1, 1, 2, 2]
		elif num_speaker == 3:
			speaker_ids = [1, 2, 3, random.choice([1,2,3])]
		else:
			speaker_ids = [1, 2, 3, 4]
		random.shuffle(speaker_ids)
		return speaker_ids
	
	def load_rs(self, file, speaker_ids, start, stop):
		ref_speech, _ = soundfile.read(self.train_path + '/target_audio_dir/' + file + '/all.wav', start = start * 640, stop = stop * 640 + 240) # Since 25 * 640 = 16000
		ref_speech = torch.FloatTensor(numpy.array(ref_speech))
		labels = []
		for speaker_id in speaker_ids:
			full_label_id = file + '_' + str(speaker_id)
			label = self.label_dic[full_label_id]
			labels.append(label[start:stop]) # Obatin the labels for the reference speech
		labels = numpy.array(labels) # 4, T
		return ref_speech, labels
	
	def load_ts(self, file, speaker_ids):
		target_speeches = []
		for speaker_id in speaker_ids:
			path = self.train_path + '/ecapa_feature_dir/' + file + '/' + str(speaker_id) + '.pt'
			feature = torch.load(path, map_location=torch.device('cpu'))
			feature = feature[random.randint(0,feature.shape[0]-1),:]
			target_speeches.append(feature)
		target_speeches = torch.stack(target_speeches) # 4, 192
		return target_speeches
	
	def __len__(self):
		return len(self.data_list)

class eval_loader(object):
	def __init__(self, eval_list, eval_path, rs_len, test_shift, **kwargs):
		self.eval_path = eval_path
		self.rs_len = int(rs_len * 25)

		self.data_list = []
		self.label_dic = defaultdict(list)

		lines = open(eval_list).read().splitlines()
		filename_set = set()
		for line in lines:
			dict = json.loads(line)
			length = len(dict['labels']) # Number of frames (1s = 25 frames)
			filename = dict['filename']
			labels = dict['labels']
			speaker_id = str(dict['speaker_key'])
			full_id = filename + '_' + speaker_id
			self.label_dic[full_id] = labels
			if filename in filename_set:
				pass
			else:
				filename_set.add(filename)
				dis = int(test_shift * 25)
				for start in range(0, length - self.rs_len, dis):
					folder = self.eval_path + '/target_audio_dir/' + filename + '/*.wav'
					audios = glob.glob(folder)
					num_speaker = len(audios) - 1
					data_intro = [filename, num_speaker, start, start + self.rs_len]
					self.data_list.append(data_intro)

	def __getitem__(self, index):
		file, num_speaker, start, stop = self.data_list[index]
		speaker_ids = self.get_ids(num_speaker)
		ref_speech, labels = self.load_rs(file, speaker_ids, start, stop)
		target_speech = self.load_ts(file, speaker_ids)
		return ref_speech, target_speech, labels, file, numpy.array(speaker_ids), numpy.array(start)
	
	def get_ids(self, num_speaker):
		speaker_ids = []
		if num_speaker == 2:
			speaker_ids = [1, 2, 1, 2]
		elif num_speaker == 3:
			speaker_ids = [1, 2, 3, 1]
		else:
			speaker_ids = [1, 2, 3, 4]
		return speaker_ids
	
	def load_rs(self, file, speaker_ids, start, stop):
		ref_speech, _ = soundfile.read(self.eval_path + '/target_audio_dir/' + file + '/all.wav', start = start * 640, stop = stop * 640 + 240)		
		ref_speech = torch.FloatTensor(numpy.array(ref_speech))
		labels = []
		for speaker_id in speaker_ids:
			full_label_id = file + '_' + str(speaker_id)
			label = self.label_dic[full_label_id]
			label_here = label[start:stop]
			# if len(label_here) < stop - start:
			# 	label_here = [0] * (stop - start)
			labels.append(label_here)
		labels = numpy.array(labels)
		return ref_speech, labels
	
	def load_ts(self, file, speaker_ids):
		target_speeches = []
		for speaker_id in speaker_ids:
			path = self.eval_path + '/ecapa_feature_dir/' + file + '/' + str(speaker_id) + '.pt'
			feature = torch.load(path, map_location=torch.device('cpu'))
			feature = torch.mean(feature, dim = 0)
			target_speeches.append(feature)
		target_speeches = torch.stack(target_speeches)
		return target_speeches
	
	def __len__(self):
		return len(self.data_list)

