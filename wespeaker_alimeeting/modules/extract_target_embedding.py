import torch, sys, os, tqdm, numpy, soundfile, time, argparse, glob, random, scipy, signal, wave
import torch.nn as nn
from speaker_encoder import ECAPA_TDNN

def init_speaker_encoder(source):
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
	batch = torch.stack(batch)    
	with torch.no_grad():
		embeddings = model.forward(batch.cuda())
	return embeddings

def get_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--target_audio_path', help='the path for the audio')
	parser.add_argument('--target_embedding_path', help='the path for the output embeddings')	
	parser.add_argument('--source', help='the part for the speaker encoder')
	parser.add_argument('--length_embedding', type=float, default=6, help='length of embeddings, seconds')
	parser.add_argument('--step_embedding', type=float, default=1, help='step of embeddings, seconds')
	parser.add_argument('--batch_size', type=int, default=96, help='step of embeddings, seconds')
	args = parser.parse_args()

	return args

def main():
	args = get_args()
	files = sorted(glob.glob(args.target_audio_path + "/*/*.wav"))
	model = init_speaker_encoder(args.source)
	for file in tqdm.tqdm(files):
		if 'all' not in file:
			batch = []
			embeddings = []
			wav_length = wave.open(file, 'rb').getnframes() # entire length for target speech
			
			if (wav_length - int(args.length_embedding * 16000)) <= 0:
				# set embedding to torch.Size([96, 192])
				# files_with_zero_length += 1
				embedding = torch.zeros((96, 192))
				embeddings.extend(embedding)
				embeddings = torch.stack(embeddings)
				output_file = args.target_embedding_path + '/' + file.split('/')[-2] + '/' + file.split('/')[-1].replace('.wav', '.pt')
				os.makedirs(os.path.dirname(output_file), exist_ok = True)
				torch.save(embeddings, output_file)
				continue

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
			embeddings = torch.stack(embeddings)
			output_file = args.target_embedding_path + '/' + file.split('/')[-2] + '/' + file.split('/')[-1].replace('.wav', '.pt')
			os.makedirs(os.path.dirname(output_file), exist_ok = True)
			torch.save(embeddings, output_file)
					
if __name__ == '__main__':
	main()