import os
import shutil

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.io.wavfile import write

fs = 22050
seconds = 3


# record sample according to parameters
print("Starting recording.\n")
myrecording = sd.rec(int((fs*seconds)), samplerate=fs, channels=2)
sd.wait()  # wait for the sample to be done recording
print("Recording done\n")
os.mkdir("samples")
os.chdir("samples")
write("sample.wav", fs, myrecording)

# load the sample and create spectrogram
y, sr = librosa.load("sample.wav", duration=3)
mels = librosa.feature.melspectrogram(y=y, sr=sr)
fig = plt.Figure()
canvas = FigureCanvas(fig)
p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
plt.savefig('melspectrogram_sample.png')

# load the image and model and make a prediction
class_labels = ['blues', 'classical', 'country', 'disco',
                'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
image_data = load_img('melspectrogram_sample.png',
                      color_mode='rgba', target_size=(288, 432))
image = img_to_array(image_data)
image = np.reshape(image, (1, 288, 432, 4))
os.chdir("..")
model = load_model("spectrogram-model")
prediction = model.predict(image/255)
prediction = prediction.reshape((10,))
class_label = np.argmax(prediction)
print(class_labels[class_label])

# delete samples folder so it wont clutter up
shutil.rmtree("samples")
