import os
import random
import shutil

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pydub import AudioSegment

# PREREQUISITE - have to create main dir and sub dirs to put audio files according to genres -> e.g. genre_samples(main dir): blues(sub dir with audio files), classical, ...

# loop through all the genres and split the audio files in 10 segments to expand the dataset
i = 0
genres = 'blues classical country disco pop hiphop jazz metal reggae rock'
genres = genres.split()
for g in genres:
    j = 0
    for filename in os.listdir(os.path.join('genre_samples', g)):

        song = os.path.join('genre_samples', g, filename)
        j = j+1
        for w in range(0, 10):
            # split track into 10 segments of 3 sec
            i = i+1
            t1 = 3*(w)*1000
            t2 = 3*(w+1)*1000
            newAudio = AudioSegment.from_wav(song)
            new = newAudio[t1:t2]
            new.export(f'audio3sec/{g}/{g+str(j)+str(w)}.wav', format="wav")


for g in genres:
    j = 0
    print(g)
    for filename in os.listdir(os.path.join('audio3sec', f"{g}")):
        song = os.path.join('audio3sec', g, filename)
        j = j+1
        # generate mel spectrograms for every audio file and save in appropriate location
        y, sr = librosa.load(song, duration=3)
        mels = librosa.feature.melspectrogram(y=y, sr=sr)

        fig = Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
        plt.savefig(f'spectrograms3sec/{g}/{g+str(j)}.png')


# randomise and move 10% of train files to test directory for validation
directory = "spectrograms3sec/train/"
for g in genres:
    filenames = os.listdir(os.path.join(directory, f"{g}"))
    random.shuffle(filenames)
    test_files = filenames[0:100]

    for f in test_files:

        shutil.move(directory + f"{g}" + "/" + f,
                    "spectrograms3sec/test/" + f"{g}")
