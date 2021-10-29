import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import os
import parselmouth
from pydub import  AudioSegment
from torchaudio import functional as F
import torchaudio
from sklearn.preprocessing import MinMaxScaler

#Functions to extract speech utterances and intonation contours and to plot them with their acoustic features

#Segment speech utterances based on
def extract_speech_utterances(dir_path, slice_path):
	files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
	pitches = [parselmouth.Sound(dir_path+f).to_pitch() for f in files ]
	fqs = [pitch.selected_array['frequency'] for pitch in pitches]
	k = 0
	for fq in fqs:
		nonzero = fq.nonzero()[0]
		diff = np.diff(nonzero)
		skip_inds = np.where(diff>1)[0]
		newInd = nonzero[0]
		filename = files[k].replace('.wav', '')
		for s in skip_inds:
			try:
				dist  = pitches[k].get_time_from_frame_number(nonzero[s+1])- pitches[k].get_time_from_frame_number(nonzero[s])
				if dist >= 0.25:
					slice_audio(pitches[k].get_time_from_frame_number(newInd), pitches[k].get_time_from_frame_number(nonzero[s]), slice_path, filename+'_'+str(s)+'.wav', dir_path+filename+'.wav')
					newInd = nonzero[s+1]
			except:
				print("error")
		dist = pitches[k].get_time_from_frame_number(len(pitches[k])) - pitches[k].get_time_from_frame_number(nonzero[-1])
		if dist >= 0.25 and dist<0.5:
			slice_audio(pitches[k].get_time_from_frame_number(nonzero[-1]), pitches[k].get_time_from_frame_number(len(pitches[k])), slice_path, filename+'_'+str(s+1)+'.wav', dir_path+filename+'.wav')
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


#extract f0 from Kaldi pitch function
def get_f0_kaldi(audio_dir):
	files = [torchaudio.load(audio_dir+f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
	pitch_feat = [ F.compute_kaldi_pitch(t[0], t[1], min_f0=50, max_f0=500)[0].tolist() for t in files]
	fqs = []
	for p in pitch_feat:
		fq = []
		for freq in p:
			fq.append(freq[1])
		fqs.append(fq)
	return fqs, files


#extract f0 from Parselmouth Praat function
def get_f0_praat(audio_dir):
	files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
	pitches = [parselmouth.Sound(audio_dir + f).to_pitch() for f in files]
	fqs = [pitch.selected_array['frequency'] for pitch in pitches]
	return fqs, files


#return a list of intervallic distances between F0 points expressed in cents
def get_interval_contour(fqs):
	contours = []
	inds=[]
	for f in fqs:
		contour = []
		ind = []
		for i in range(len(f)-1):
			if i < len(f):
				if f[i] == 0 or f[i+1] == 0:
					pass
				else:
					dist = 1200 * np.log2(f[i+1]/f[i])
					contour.append(dist)
					ind.append((i, i+1))
		contours.append(contour)
		inds.append(ind)
	return contours, inds


#Get contour approximation by intervals in cents
def create_contour_approximation(conts):
	approx=[]
	for lista in conts:
		app=[]
		carry = 0
		for inter in lista:
			i = abs(inter)
			if i<100 and carry<100:
				app.append('0')
				carry += i
			elif i or carry >=100 and i or carry <200:
				if inter<0:
					app.append('-1')
					carry = 0
				else:
					app.append('1')
					carry = 0
			elif i or carry >=200 and i or carry <300:
				if inter<0:
					app.append('-2')
					carry = 0
				else:
					app.append('2')
					carry = 0
			elif i or carry >=300 and i or carry <400:
				if inter<0:
					app.append('-3')
					carry = 0
				else:
					app.append('3')
					carry = 0
			elif i or carry >=400 and i or carry <500:
				if inter<0:
					app.append('-4')
					carry = 0
				else:
					app.append('4')
					carry = 0
			elif i or carry >=500 and i or carry <600:
				if inter<0:
					app.append('-5')
					carry = 0
				else:
					app.append('5')
					carry =0
			elif i or carry >=600 and i or carry <700:
				if inter<0:
					app.append('-6')
					carry = 0
				else:
					app.append('6')
					carry =0
			elif i or carry >=700 and i or carry <800:
				if inter<0:
					app.append('-7')
					carry = 0
				else:
					app.append('7')
					carry =0
			elif i or carry >=800 and i or carry <900:
				if inter<0:
					app.append('-8')
					carry = 0
				else:
					app.append('8')
					carry =0
			elif i or carry >900:
				if inter<0:
					app.append('-10')
					carry = 0
				else:
					app.append('10')
					carry =0
		approx.append(app)
	return approx


def get_contour_scale(fq):
	fq = fq.reshape(-1, 1)
	scaler = MinMaxScaler((0, 5))
	scaler.fit(fq)
	contour = scaler.transform(fq)
	return contour


def get_contour_kmeans(f0, n_clusters):
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