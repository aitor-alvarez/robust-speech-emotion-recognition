from utils.intonation import *
from utils.gapbide import Gapbide
from utils.ClosedPatterns import ClosedPatterns
from statistics import mean, stdev
import pandas as pd


def get_patterns(audio_dir, emotion='neutral'):
	#extract f0 and get the contours for each audio file
	fqs, files = get_f0_praat(audio_dir)
	fq_list = [[f for f in fq if f>0] for fq in fqs]
	F0_mean =[mean(fqs) for fqs in fq_list]
	F0_range = [max(fqs) - min(fqs) for fqs in fq_list]
	contours, inds = get_interval_contour(fqs)
	intervals, inds = extract_contour_slope(contours)
	pk_to_pk, val_to_val, pos_slope, neg_slope = peak_to_peak(intervals, inds)
	stdv_pk = [stdev(p) for p in pk_to_pk if len(p)>1]
	stdv_val = [stdev(v) for v in val_to_val if len(v)>1]
	stdv_pos_slope = [stdev(p) for p in pos_slope if len(p)>1]
	stdv_neg_slope = [stdev(n) for n in neg_slope if len(n)>1]
	pattern_length = 4
	intervals_file = 'patterns/'+emotion
	g1 = Gapbide(intervals, 4, 0, 0, pattern_length, intervals_file)
	g1.run()
	pats = ClosedPatterns(intervals_file+'_intervals.txt', intervals_file+'_maximal.txt')
	pats.execute()
	data1 = pd.DataFrame({'F0_mean': F0_mean})
	data2 = pd.DataFrame({'F0_range': F0_range})
	data3 = pd.DataFrame({'stdv_pk ': stdv_pk})
	data4 = pd.DataFrame({'stdv_val':stdv_val})
	data5 = pd.DataFrame({'stdv_pos_slope': stdv_pos_slope})
	data6 = pd.DataFrame({'stdv_neg_slope': stdv_neg_slope })
	df = pd.concat([data1, data2, data3, data4, data5, data6], axis=1)
	df.to_excel('results_'+emotion+'.xlsx')




