from utils.intonation import *


def get_contour_approximation(audio_dir):
	#extract f0 and get the contours for each audio file
	fqs, files = get_f0_praat(audio_dir)
	contours, inds = get_interval_contour(fqs)
	intervals = get_interval_representation(contours)





