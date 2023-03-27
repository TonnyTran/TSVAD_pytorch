import glob, numpy, os, random, soundfile, torch, wave, json
from collections import defaultdict
from scipy import signal
import torchaudio.compliance.kaldi as kaldi

def init_loader(args):
	trainloader = train_loader(**vars(args))
	args.trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
	evalLoader = eval_loader(**vars(args))
	args.evalLoader = torch.utils.data.DataLoader(evalLoader, batch_size = args.batch_size, shuffle = False, num_workers = args.n_cpu, drop_last = False)
	return args

class train_loader(object):
	def __init__(self, train_list, train_path, ts_len, rs_len, musan_path, rir_path, epoch, warm_up_epoch, **kwargs):
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

		self.epoch = epoch
		self.warm_up_epoch = warm_up_epoch
		self.train_path = train_path
		self.ts_len = int(ts_len / 40) # Number of frames for target speech
		self.rs_len = int(rs_len / 40) # Number of frames for reference speech

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
				dis = 25 * 6
				for start in range(0, length - self.rs_len, dis):
					folder = self.train_path + '/' + filename + '/*.wav'
					audios = glob.glob(folder)
					num_speaker = len(audios) - 1 # The total number of speakers, 2 or 3 or 4
					data_intro = [filename, num_speaker, start, start + self.rs_len]
					self.data_list.append(data_intro)

	def __getitem__(self, index):
		# T: number of frames (1 frame = 0.04s)
		# ref_speech : 16000 * (T / 25)
		# labels : 4, T
		# target_speech: 4, 16000 * (T / 25)
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
		ref_speech, _ = soundfile.read(os.path.join(self.train_path, file + '/all.wav'), start = start * 640, stop = stop * 640 + 240) # Since 25 * 640 = 16000
		# soundfile.write('ref_speech.wav', ref_speech, 16000)

		frame_len = int(self.rs_len / 25 * 16000) + 240
		ref_speech = numpy.expand_dims(numpy.array(ref_speech), axis = 0)
		augtype = random.randint(0,4)
		if self.epoch < self.warm_up_epoch:
			augtype = 0
		if augtype == 0:
			ref_speech = ref_speech
		elif augtype == 1:
			ref_speech = self.add_rev(ref_speech, length = frame_len)
		elif augtype == 2:
			ref_speech = self.add_noise(ref_speech, 'speech', length = frame_len)
		elif augtype == 3: 
			ref_speech = self.add_noise(ref_speech, 'music', length = frame_len)
		elif augtype == 4:
			ref_speech = self.add_noise(ref_speech, 'noise', length = frame_len)
		ref_speech = ref_speech[0]
		
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
			path = os.path.join(self.train_path, file, str(speaker_id) + '.wav')
			wav_length = wave.open(path, 'rb').getnframes() # entire length for target speech
			start = numpy.int64(random.random()*(wav_length-int(self.ts_len / 25 * 16000) - 240)) # start point
			frame_len = int(self.ts_len / 25 * 16000) + 240
			stop = start + frame_len
			target_speech, _ = soundfile.read(path, start = start, stop = stop)

			target_speech = numpy.expand_dims(numpy.array(target_speech), axis = 0)
			augtype = random.randint(0,4)
			if self.epoch < self.warm_up_epoch:
				augtype = 0
			if augtype == 0:
				target_speech = target_speech
			elif augtype == 1:
				target_speech = self.add_rev(target_speech, length = frame_len)
			elif augtype == 2:
				target_speech = self.add_noise(target_speech, 'speech', length = frame_len)
			elif augtype == 3: 
				target_speech = self.add_noise(target_speech, 'music', length = frame_len)
			elif augtype == 4:
				target_speech = self.add_noise(target_speech, 'noise', length = frame_len)
			target_speech = target_speech[0]
		
			target_speech = torch.FloatTensor(numpy.array(target_speech))
			target_speech = (target_speech * (1 << 15)).unsqueeze(0)			
			target_speech = kaldi.fbank(target_speech, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0, sample_frequency=16000, window_type='hamming', use_energy=False)
			target_speech = torch.permute(target_speech, (1, 0))
			# soundfile.write(str(speaker_id) + '.wav', target_speech, 16000) # For debug
			target_speeches.append(target_speech)
		target_speeches = torch.stack(target_speeches) # 4, 16000 * (T / 25)
		return target_speeches
	
	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio, length):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:length]

	def add_noise(self, audio, noisecat, length):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiselength = wave.open(noise, 'rb').getnframes()
			if noiselength <= length:
				noiseaudio, _ = soundfile.read(noise)
				noiseaudio = numpy.pad(noiseaudio, (0, length - noiselength), 'wrap')
			else:
				start_frame = numpy.int64(random.random()*(noiselength-length))
				noiseaudio, _ = soundfile.read(noise, start = start_frame, stop = start_frame + length)
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio
	
class eval_loader(object):
	def __init__(self, eval_list, eval_path, ts_len, rs_len, **kwargs):
		self.eval_path = eval_path
		self.ts_len = int(ts_len / 40)
		self.rs_len = int(rs_len / 40)

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
				dis = self.rs_len
				for start in range(0, length - self.rs_len, dis):
					folder = self.eval_path + '/' + filename + '/*.wav'
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
		ref_speech, _ = soundfile.read(os.path.join(self.eval_path, file + '/all.wav'), start = start * 640, stop = stop * 640 + 240)		
		ref_speech = torch.FloatTensor(numpy.array(ref_speech))
		labels = []
		for speaker_id in speaker_ids:
			full_label_id = file + '_' + str(speaker_id)
			label = self.label_dic[full_label_id]
			labels.append(label[start:stop])
		labels = numpy.array(labels)
		return ref_speech, labels
	
	def load_ts(self, file, speaker_ids):
		target_speeches = []
		for speaker_id in speaker_ids:
			path = os.path.join(self.eval_path, file, str(speaker_id) + '.wav')
			start = 0
			stop = start + int(self.ts_len / 40 * 16000) + 240
			target_speech, _ = soundfile.read(path, start = start, stop = stop)
			target_speech = torch.FloatTensor(numpy.array(target_speech))
			target_speech = (target_speech * (1 << 15)).unsqueeze(0)			
			target_speech = kaldi.fbank(target_speech, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0, sample_frequency=16000, window_type='hamming', use_energy=False)
			target_speech = torch.permute(target_speech, (1, 0))
			target_speeches.append(target_speech)
		target_speeches = torch.stack(target_speeches)
		return target_speeches
	
	def __len__(self):
		return len(self.data_list)

