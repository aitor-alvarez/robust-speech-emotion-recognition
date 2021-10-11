import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import os
import parselmouth
from pydub import  AudioSegment
import uuid
from sklearn.preprocessing import MinMaxScaler

#Functions to extract speech utterances and intonation contours and to plot them with their acoustic features


def extract_speech_utterances(dir_path, slice_path):
	files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
	pitches = [parselmouth.Sound(dir_path+f).to_pitch() for f in files ]
	fqs = [pitch.selected_array['frequency'] for pitch in pitches]
	k = 0
	for fq in fqs:
		print(k)
		nonzero = fq.nonzero()[0]
		diff = np.diff(nonzero)
		skip_inds = np.where(diff>1)[0]
		newInd = nonzero[0]
		filename = files[k].replace('.wav', '')
		for s in skip_inds:
			try:
				dist  = pitches[k].get_time_from_frame_number(nonzero[s+1])- pitches[k].get_time_from_frame_number(nonzero[s])
				if dist >= 0.25:
					slice_audio(pitches[k].get_time_from_frame_number(newInd), pitches[k].get_time_from_frame_number(nonzero[s]), slice_path, filename+'_'+str(uuid.uuid4())+'.wav', dir_path+filename+'.wav')
					newInd = nonzero[s+1]
			except:
				print("error")
		dist = pitches[k].get_time_from_frame_number(len(pitches[k])) - pitches[k].get_time_from_frame_number(nonzero[-1])
		if dist >= 0.25 and dist<0.5:
			slice_audio(pitches[k].get_time_from_frame_number(nonzero[-1]), pitches[k].get_time_from_frame_number(len(pitches[k])), slice_path, filename+'_'+str(uuid.uuid4())+'.wav', dir_path+filename+'.wav')
		k +=1
	print("segmentation completed")


def slice_audio(slice_from, slice_to, path, name, audio_file):
	audio = AudioSegment.from_wav(audio_file)
	try:
		seg = audio[slice_from * 1000:slice_to * 1100]
		seg.set_channels(2)
		seg.export(path+name, format="wav", bitrate="192k")
	except:
		print("NO")


def get_contour(fq):
	fq = fq.reshape(-1, 1)
	scaler = MinMaxScaler((0, 5))
	scaler.fit(fq)
	contour = scaler.transform(fq)
	return contour


def get_contour2(f0, n_clusters):
	f0 = f0.reshape(-1, 1)
	scores=[]
	c =[]
	for n in range(2, n_clusters):
		model = KMeans(n_clusters=n, random_state=10)
		groups = model.fit_predict(f0)
		avg_score = silhouette_score(f0, groups)
		scores.append(avg_score)
		c.append(n)
		print("For clusters =", n,
					"The average silhouette_score is :", avg_score)
	index = scores.index(max(scores))
	model = KMeans(n_clusters=c[index], random_state=10)
	contour = model.fit_predict(f0.reshape(-1, 1))
	return contour


def plot_pitch(pitch):
	pitch_values = pitch.selected_array['frequency']
	pitch_values[pitch_values == 0] = np.nan
	plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
	plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
	plt.grid(False)
	plt.ylim(0, pitch.ceiling)
	plt.ylabel("fundamental frequency [Hz]")


def plot_spectrogram(spectrogram, dynamic_range=70):
		X, Y = spectrogram.x_grid(), spectrogram.y_grid()
		sg_db = 10 * np.log10(spectrogram.values)
		plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
		plt.ylim([spectrogram.ymin, spectrogram.ymax])
		plt.xlabel("time [s]")
		plt.ylabel("frequency [Hz]")


def plot_intensity(intensity):
	plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
	plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
	plt.grid(False)
	plt.ylim(0)
	plt.ylabel("intensity [dB]")