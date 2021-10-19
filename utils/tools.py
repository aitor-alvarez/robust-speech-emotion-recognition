from tinytag import TinyTag
import pandas as pd
import os

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