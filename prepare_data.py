import glob, tqdm, os, textgrid, soundfile, copy, json
from collections import defaultdict

class Segment(object):
    def __init__(self, uttid, spkr, stime, etime, text):
        self.uttid = uttid
        self.spkr = spkr
        self.stime = round(stime, 2)
        self.etime = round(etime, 2)
        self.text = text

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

path = '/data08/alimeeting' # The path of alimeeting
type = 'Train' # Train then Eval, output is ts_Train.json. ts_Eval.json, target_audio_dir folder for Train and Eval
path_train = os.path.join(path, '%s_Ali_far'%(type))
path_train_target_wav = os.path.join(path_train, 'target_audio_dir')
path_train_wav = os.path.join(path_train, 'audio_dir')
path_train_grid = os.path.join(path_train, 'textgrid_dir')
out_train_text = os.path.join(path, 'ts_%s.json'%(type))

text_grids = glob.glob(path_train_grid + '/*')
outs = open(out_train_text, "w")
for text_grid in tqdm.tqdm(text_grids):
    tg = textgrid.TextGrid.fromFile(text_grid)
    segments = []
    spk = {}
    num_spk = 1
    uttid = text_grid.split('/')[-1][:-9]
    for i in range(tg.__len__()):
        for j in range(tg[i].__len__()):
            if tg[i][j].mark:
                if tg[i].name not in spk:
                    spk[tg[i].name] = num_spk
                    num_spk += 1
                segments.append(Segment(
                        uttid,
                        spk[tg[i].name],
                        tg[i][j].minTime,
                        tg[i][j].maxTime,
                        tg[i][j].mark.strip(),
                    )
                )
    segments = sorted(segments, key=lambda x: x.spkr)

    no_overlap_segments = []

    intervals = defaultdict(list)
    new_intervals = defaultdict(list)

    # Summary the intervals for all speakers
    for i in range(len(segments)):
        interval = [segments[i].stime, segments[i].etime]
        intervals[segments[i].spkr].append(interval)

    # Remove the overlapped speeech    
    for key in intervals:
        new_interval = intervals[key]
        for o_key in intervals:
            if o_key != key:                
                new_interval = remove_overlap(copy.deepcopy(new_interval), copy.deepcopy(intervals[o_key]))
        new_intervals[key] = new_interval

    wav_file = glob.glob(os.path.join(path_train_wav, uttid) + '*.wav')[0]
    orig_audio, fs = soundfile.read(wav_file)
    orig_audio = orig_audio[:,0]
    length = len(orig_audio) 

    # # Cut and save the clean speech part
    id_full = wav_file.split('/')[-1][:-4]
    for key in new_intervals:
        output_dir = os.path.join(path_train_target_wav, id_full)
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
        
        res = {'filename':id_full, 'speaker_key':key, 'labels':labels}
        soundfile.write(output_wav, new_audio, 16000)
    output_wav = os.path.join(output_dir, 'all.wav')
    soundfile.write(output_wav, orig_audio, 16000)

    # Save the labels
    for key in intervals:
        labels = [0] * int(length / 16000 * 25) # 40ms, one label        
        for interval in intervals[key]:
            s, e = interval
            for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
                labels[i] = 1
        
        res = {'filename':id_full, 'speaker_key':key, 'labels':labels}
        json.dump(res, outs)
        outs.write('\n')
