from flask import Blueprint, render_template, request
from werkzeug.utils import secure_filename
import os
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from mutagen.wave import WAVE
import pickle
import librosa
import numpy as np
from collections import Counter
from scipy.io import wavfile
import pandas as pd
from tqdm import tqdm
import librosa
from librosa.feature import mfcc
import keras

prediction = Blueprint('prediction',__name__)
#feature extraction
def get_mfcc(path):
    audio, sr = librosa.load(path)
    mfccs = mfcc(y=audio,sr=sr,n_mfcc=13,hop_length=512,n_fft=2048)
    return mfccs
#divide into frames
def get_frame_mfccs(path):
    '''
    -------------------------------------------------
    Loads the .wav audio file and split into 3-second slices and then calculate mfccs for all slices
    -------------------------------------------------
    :params path : path to .wav file
    :returns : list of mfcc values'''
    audio, sr = librosa.load(path)
    frames = librosa.util.frame(audio, frame_length=sr*3, hop_length=sr*3)
    frame_mfccs = []
    for i in range(frames.shape[1]):
        mfccs = mfcc(y=frames[:,i],sr=sr,n_mfcc=13,hop_length=512,n_fft=2048)
        frame_mfccs.append(mfccs)
    return frame_mfccs
def reshape(data,shape=(26,65)):
    assert data.shape == (13,130) , f"The Data shape should be (13,130) but got {data.shape}"
    data = data.reshape(shape)
    data = np.expand_dims(data,axis=-1)
    return data

#C:\Users\sreer\Flask Web Development\website\audio\test
#C:\Users\sreer\Flask Web Development\website\audio\wave

def cnn_predict():

    path = "C://Users//sreer//Flask Web Development//website//audio//wave"
    model = keras.models.load_model('website\models\cnn_indian1.h5')
    classes = {0:'carnatic', 1:'folk', 2:'ghazal', 3:'semiclassical', 4:'sufi'}
    dir_list = os.listdir(path)
    res = []
    for input_file in dir_list:
        
        mfccss = get_frame_mfccs(f"website/audio/wave/{input_file}")
        for frame in mfccss:
            res.append(frame)

    processed_res = np.array([reshape(x) for x in res])

    pred = model.predict(processed_res)
    preds = []
    for i in pred:
        out1 = np.argmax(i)
        preds.append(out1)
    preds2 = []
    for pred in preds:
        preds2.append(classes[pred])
    
    preds2 = Counter(preds2)
    res = max(preds2,key=preds2.get)
    print(preds2)
    print(res)
    return res
    

def preprocess():
    
    #Wave Conversion
    path = "C://Users//sreer//Flask Web Development//website//audio//test"
    # path = "./audio/test"
    dir_list = os.listdir(path)
    print(dir_list)
    input_file=dir_list[0]
    print(input_file)
    try:
        output_file = f"website/audio/wave/{input_file[:-4]}.wav"
        sound = AudioSegment.from_mp3(f"website/audio/test/{input_file}")
        sound.export(output_file, format="wav")
        print('Successfully Converted: ', input_file)
    except CouldntDecodeError:
        print('Couldnt convert the file: ', input_file) 

#C:\Users\sreer\Flask Web Development\website\audio\test

    #Deleting files

    # path = "C://Users//thash//Flask Web Development//website//audio//cut"
    # dir_list = os.listdir(path)
    # for input_file in dir_list:
    #     os.remove('website/audio/cut/'+input_file)
    
    #Trimming
    # print("conversion done\n\n\n")
    # path = "C://Users//thash//Flask Web Development//website//audio//wave"
    # dir_list = os.listdir(path)
    # input_file=dir_list[0]
    # duration = 3
    # duration_mil = duration*1000
    # song = AudioSegment.from_file(f"website/audio/wave/{input_file}",
    #                                 format="wav")
    # audio = WAVE(f"website/audio/wave/{input_file}")
    # n=1
    # #cuttinng starts
    # for i in range(0,int(audio.info.length)//duration):

    #     thirty_sec = song[i*duration_mil : (i+1)*duration_mil]
    #     thirty_sec.export(f"website/audio/cut/{input_file[:-4]}{n}.wav", format="wav")
    #     n+=1
        
    

    # if int(audio.info.length)%duration != 0:
    #     thirty_sec = song[-duration_mil:]
    #     thirty_sec.export(f"website/audio/cut/{input_file[:-4]}{n}.wav", format="wav")
    #     n += 1

    # print("trimming done\n\n\n")
    
    

    
@prediction.route('/indian')
def indian():
    return render_template('indian.html')
@prediction.route('/western')
def western():
    return render_template('western.html')

@prediction.route('/indian',methods=['POST'])
def predict():
    audiofile=request.files['audiofile']
    filename=audiofile.filename
    audiofile.save("website/audio/test/test.mp3")
    preprocess()
    res = cnn_predict()

    
    return render_template('indian.html', predicted_genre = res, filename=filename)
 