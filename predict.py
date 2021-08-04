import os
import sounddevice as sd
from scipy.io.wavfile import write
import shutil


fs = 22050
seconds = 30

# record sample according to parameters
myrecording = sd.rec(int((fs*seconds)), samplerate=fs, channels=2)
sd.wait() # wait for the sample to be done recording
print("Recording done\n")
os.mkdir("samples")
os.chdir("samples")
write("sample.wav",fs,myrecording)


# analysis and predict goes here

os.chdir('../')
shutil.rmtree("samples")