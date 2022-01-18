from tinytag import TinyTag
import pandas as pd
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt



def write_metadata_to_excel(audio_dir):
	file_name =[]
	speaker=[]
	gender=[]
	for f in os.listdir(audio_dir):
		if f.endswith('.wav'):
			meta = TinyTag.get(audio_dir+f)
			file_name.append(f)
			speaker.append(meta.artist)
			gender.append(meta.title)
	data={'file_name': file_name, 'speaker': speaker, 'gender': gender}
	df = pd.DataFrame(data)
	df.to_excel('speaker_data.xlsx')


def get_activation_values(files, data_file, file_field, activation_field):
	df = pd.read_excel(data_file, engine='openpyxl')
	activations = []
	for f in files:
		index = df.where(df[file_field] == f).index.min()
		activations.append(df[activation_field][index])
	return activations


def extract_audio_from_video(video_dir, output_dir):
	for f in os.listdir(video_dir):
		if f.endswith('.mp4'):
			audio = AudioSegment.from_file(video_dir+f)
			audio.export(output_dir+f.replace('mp4', 'wav'), format="wav")


#Careful! deletes files that are not in spreadsheet!
def cleanup_directory(excel_file, dir):
	df = pd.read_excel(excel_file, engine='openpyxl')
	for root, dirs, files in os.walk(dir):
		path = root.split(os.sep)
		for file in files:
			if file.endswith('.wav'):
				a = df['wav_file'] == file
				if a.any():
					pass
				else:
					os.remove(dir+path[-1]+'/'+file)


def plot_serie(serie, index, emotion=None):
	plt.style.use('seaborn')
	plt.plot(index, serie, marker='o', color='dimgrey', linestyle='solid')
	plt.tight_layout()
	plt.title(emotion)
	plt.xlabel('Time steps (50 ms)')
	plt.ylabel('Interval')
	plt.show()

