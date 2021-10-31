from utils.intonation import *
from utils.gapbide import Gapbide
from utils.ClosedPatterns import ClosedPatterns


def get_patterns(audio_dir, emotion='neutral'):
	#extract f0 and get the contours for each audio file
	fqs, files = get_f0_praat(audio_dir)
	contours, inds = get_interval_contour(fqs)
	intervals, inds = extract_contour_slope(contours)
	pattern_length = 3
	intervals_file = 'patterns/'+emotion
	g1 = Gapbide(intervals, 2, 0, 0, pattern_length, intervals_file)
	g1.run()
	pats = ClosedPatterns(intervals_file+'_intervals.txt')
	maximal = pats.execute()




