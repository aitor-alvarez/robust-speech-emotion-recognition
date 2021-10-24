from tinytag import TinyTag
import pandas as pd
import os
from pydub import AudioSegment

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


def extract_audio_from_video(video_dir, output_dir):
	for f in os.listdir(video_dir):
		if f.endswith('.mp4'):
			audio = AudioSegment.from_file(video_dir+f)
			audio.export(output_dir+f.replace('mp4', 'wav'), format="wav")

