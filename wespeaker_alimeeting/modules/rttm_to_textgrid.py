# Import necessary libraries 
import glob, tqdm, os, textgrid, soundfile, copy, json, argparse, numpy, torch, wave


def get_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--input_rttm_file', help='the path for the dihard rttm file')
	parser.add_argument('--output_textgrid_file', help='the path for the dihard rttm file')

	args = parser.parse_args()
	return args

def main():
    args = get_args()
    input_rttm_file = args.input_rttm_file
    output_textgrid_file = args.output_textgrid_file

    # Initialize the xmax variable
    xmax = 0
    # Set to store unique speaker IDs
    unique_speaker_ids = set()

    # Open the output TextGrid file for writing
    with open(output_textgrid_file, 'w') as textgrid_file:
        # Read the RTTM file to calculate the maximum end time (xmax)
        with open(input_rttm_file, 'r') as rttm_file:
            for line in rttm_file:
                fields = line.strip().split()
                end_time = float(fields[3]) + float(fields[4])
                xmax = max(xmax, end_time)
                unique_speaker_ids.add(fields[7])
        
        size = len(unique_speaker_ids)

        # Write the TextGrid header with calculated xmax
        textgrid_file.write('File type = "ooTextFile"\n')
        textgrid_file.write('Object class = "TextGrid"\n\n')
        textgrid_file.write('xmin = 0\n')
        textgrid_file.write(f'xmax = {xmax}\n')  # Write the calculated xmax
        textgrid_file.write('tiers? <exists>\n')
        textgrid_file.write(f'size = {size}\n')
        textgrid_file.write('item []:\n')

        # Reset the interval_id
        interval_id = 1

        # Read the RTTM file and generate IntervalTier entries
        with open(input_rttm_file, 'r') as rttm_file:
            # Dictionary to store intervals by speaker
            intervals_by_speaker = {}

            for line in rttm_file:
                # Split each line into fields
                fields = line.strip().split()

                # Extract relevant information from the RTTM fields
                speaker_id = fields[7]  # Assuming speaker ID is in field 8 (adjust if needed)
                start_time = float(fields[3])  # Assuming start time is in field 4 (adjust if needed)
                end_time = start_time + float(fields[4])  # Calculate end time
                text = ""  # Empty text field

                # Store intervals by speaker
                if speaker_id not in intervals_by_speaker:
                    intervals_by_speaker[speaker_id] = []
                intervals_by_speaker[speaker_id].append((start_time, end_time, text))

            # Write IntervalTier entries for each speaker
            for speaker_id, intervals in intervals_by_speaker.items():
                # Write IntervalTier header
                textgrid_file.write(f'\titem [{interval_id}]:\n')
                textgrid_file.write('\t\tclass = "IntervalTier"\n')
                textgrid_file.write(f'\t\tname = "{speaker_id}"\n')
                textgrid_file.write(f'\t\txmin = 0\n')
                textgrid_file.write(f'\t\txmax = {xmax}\n')
                textgrid_file.write(f'\t\tintervals: size = {len(intervals)}\n')

                # Write intervals for this speaker
                for i, (start_time, end_time, text) in enumerate(intervals, start=1):
                    textgrid_file.write(f'\t\tintervals [{i}]:\n')
                    textgrid_file.write(f'\t\t\txmin = {start_time}\n')
                    textgrid_file.write(f'\t\t\txmax = {end_time}\n')
                    textgrid_file.write(f'\t\t\ttext = "test"\n') # Writing text as test because it cannot be left empty

                # Increment the interval ID
                interval_id += 1

    print(f'Conversion complete. TextGrid file "{output_textgrid_file}" has been generated.')

if __name__ == '__main__':
    main()