import json
import math
import os

import librosa

DATASET_PATH = "genre_samples"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
SAMPLES_PER_TRACK = SAMPLE_RATE * 30  # sample rate times the sample duration


def save_data(dataset_path, json_path, n_mfcc=20, n_fft=2048, hop_length=512, num_segments=10):

    # store data
    data = {
        "genres": [],
        "mfcc_values": [],
        "labels": []
    }

    num_samples_per_segment = SAMPLES_PER_TRACK / num_segments
    expected_num_mfcc_vectors_per_segment = math.ceil(
        num_samples_per_segment / hop_length)  # ceil -> round up to int

    # loop through genres
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):

        # must not be at root dir
        if dirpath is not dataset_path:

            # save names of folders
            # split - genre_samples/rock => ["genre", "rock"] | Splits the path into words in a list
            genre = os.path.split(dirpath)[-1]  # get the last item in list
            data["genres"].append(genre)
            print("\nProcessing {}".format(genre))

            # process samples for a genre
            for f in filenames:

                # get the path of a sample and load it
                file_path = os.path.join(dirpath, f)
                sample, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments, extract mfcc and store data
                for s in range(num_segments):
                    start_point = int(num_samples_per_segment * s)
                    end_point = int(start_point + num_samples_per_segment)

                    mfcc = librosa.feature.mfcc(sample[start_point:end_point],
                                                sr=SAMPLE_RATE,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)

                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc_values"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


save_data(DATASET_PATH, JSON_PATH)
