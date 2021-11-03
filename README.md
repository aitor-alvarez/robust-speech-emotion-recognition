# robust-speech-emotion-recognition
## Speech emotion recognition models and feature extractors

For extracting tonal rhythm from utterances execute the following command:

python main.py -a "/the_directory/where_files/are_located/" -e "emotion_tag"

arguments:
#
  -h, --help            show this help message and exit
  -a AUDIO_DIR, --audio_dir AUDIO_DIR
                        The audio directory where the files are located
  -e EMOTION, --emotion EMOTION
                        Emotion tag



This command will return the results in a spreadsheet. All closed and maximal patterns will be outputed in 2 different files. Pattern files will be located in the directory "patterns".
