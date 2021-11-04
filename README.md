# Robust Speech Emotion Recognition
### Speech emotion recognition models and feature extractors

For extracting tonal rhythm from utterances execute the following command:

python main.py -a "/the_directory/where_files/are_located/" -e "emotion_tag"

Where -a is the directory of the audio files for a given emotion, and -e is the emotion tag.


This command will return the results in a spreadsheet. All closed and maximal patterns will be outputed in 2 different files. Pattern files will be located in the directory "patterns".

An extra parameter -d can be added to add activation values from an external spreadsheet.

To see all the parameters use the help function -h.
