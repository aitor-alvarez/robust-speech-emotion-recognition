from utils.intonation import *
from utils.gapbide import Gapbide
from utils.ClosedPatterns import ClosedPatterns, UniquePatterns
from statistics import mean, stdev
import pandas as pd
import argparse
from utils.tools import get_activation_values
from scipy import stats


def get_patterns(audio_dir, emotion='neutral', data_file =None):
  #extract f0 and get the contours for each audio file
  fqs, files = get_f0_praat(audio_dir)
  if data_file:
    activations = get_activation_values(files, data_file, 'wav_file', 'activation')
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
  data0 = pd.DataFrame({'Filename': files})
  data1 = pd.DataFrame({'F0_mean': stats.zscore(F0_mean)})
  data2 = pd.DataFrame({'F0_range': stats.zscore(F0_range)})
  data3 = pd.DataFrame({'stdv_pk': stdv_pk})
  data4 = pd.DataFrame({'stdv_val': stdv_val})
  data5 = pd.DataFrame({'stdv_pos_slope': stdv_pos_slope})
  data6 = pd.DataFrame({'stdv_neg_slope': stdv_neg_slope })
  if data_file:
    data00= pd.DataFrame({'Arousal': activations})
    df = pd.concat([data0, data00, data1, data2, data3, data4, data5, data6], axis=1)
    df['Macro_Rhythm_Ind'] = df['stdv_pk'] + df['stdv_val'] + df['stdv_pos_slope'] + df['stdv_neg_slope']
    df['Macro_Rhythm_Ind'] = stats.zscore(df['Macro_Rhythm_Ind'])
    df.to_excel('results_' + emotion + '.xlsx')
  else:
    df = pd.concat([data0, data1, data2, data3, data4, data5, data6], axis=1)
    df['Macro_Rhythm_Ind'] = df['stdv_pk'] + df['stdv_val'] + df['stdv_pos_slope'] + df['stdv_neg_slope']
    df['Macro_Rhythm_Ind'] = stats.zscore(df['Macro_Rhythm_Ind'])
    df.to_excel('results_'+emotion+'.xlsx')


def get_unique_patterns(file1, file2):
	uni = UniquePatterns(file1, file2)
	uni.get_unique_patterns()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--audio_dir', type=str, default = None,
                        help='The audio directory where the files are located')

    parser.add_argument('-e', '--emotion', type=str, default = None,
                        help='Emotion tag')

    parser.add_argument('-d', '--data_file', type=str, default=None,
                        help='Excel spreadsheet with arousal levels per audio file')

    parser.add_argument('-r', '--reference_file', type=str, default=None,
                        help='Text file used as a reference to be compared against')

    parser.add_argument('-c', '--compare_file', type=str, default=None,
                        help='Text file used to compared against the reference to find out unique patterns')

    args = parser.parse_args()

    if args.reference_file and args.compare_file:
      get_unique_patterns(args.reference_file, args.compare_file)

    else:
      get_patterns(args.audio_dir, args.emotion)


if __name__ == '__main__':
    main()
