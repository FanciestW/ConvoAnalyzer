import librosa
import io
import os
import speech_recognition
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sys import platform as sys_pf
from scipy.fftpack import fft

import watson_developer_cloud
from os.path import join, dirname
import json

speech_to_text = watson_developer_cloud.SpeechToTextV1(
    iam_apikey='65qVRIfF-CQYgk268h9NJERzwiY-1xfRY6WCmpU9iF2L',
    url='https://stream.watsonplatform.net/speech-to-text/api'
)


def main():
    audio_file = "audio/dailylife002.wav"
    signal,sample_rate = librosa.load(audio_file, sr=None, mono=True)

    smoothed_signal = abs(signal)

    out_dir = "out/"
    if not os.path.exists(os.path.dirname(out_dir)):
        try:
            os.makedirs(os.path.dirname(out_dir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    write_to_audio_file("out/before_smooth.wav", smoothed_signal, sample_rate)

    smoothed_signal = smooth_signal(smoothed_signal, 10, 10)

    write_to_audio_file("out/after_smooth.wav", smoothed_signal, sample_rate)

    threshold = 0.015
    silence_window = int(sample_rate / 500)

    bool_signal = get_bool_arr(smoothed_signal, threshold, silence_window)
    bool_signal = smooth_signal(bool_signal, int(sample_rate/500), 100)
    write_to_audio_file("out/before_bool.wav", bool_signal.astype(float), sample_rate)
    bool_signal = cut_audio(bool_signal, 0.150)
    write_to_audio_file("out/after_bool.wav", bool_signal.astype(float), sample_rate)

    audio_clips = split_np_array(signal, bool_signal)
    gender_arr = []
    for i, clip in enumerate(audio_clips):
        write_to_audio_file(f"out/clip{i}.wav", clip, sample_rate)
        get_gender(smooth_signal(clip, window=10, passes=10))
    getAudioText("file")

def get_gender(signal, sr=44100):
    fourier = np.fft.rfft(signal)
    fourier_mag = np.abs(fourier)
    fourier_mag = smooth_signal(fourier_mag, 10, 100)
    freq = np.fft.rfftfreq(signal.size, d=1./sr)
    start_male_range = np.argmax(freq >= 85)
    end_male_range = np.argmax(freq >= 185)
    start_female_range = np.argmax(freq >= 165)
    end_female_range = np.argmax(freq >= 255)

    male_avg = np.mean(fourier_mag[start_male_range:end_male_range])
    female_avg = np.mean(fourier_mag[start_female_range:end_female_range])
    max_freq = freq[np.argmax(fourier_mag)]
    # print(male_avg, female_avg, np.argmax(fourier_mag), max_freq)
    if male_avg > female_avg:
        print('Male')
    else:
        print('Female')
    
    # print(f"Lows: {low_count} Highs: {high_count}")
    # 
    # if max_freq < 180:
    #     print("male")
    # else:
    #     print("female")

def cut_audio(arr, tolerence=0.2, sr=44100):
    time_stamps = list()
    bool_arr = []
    
    high = False
    i = 0
    j = 0
    while i < len(arr):
        # Start of new audio segment
        if arr[i] >= 0.5 and not high:
            time_stamps.append(i / sr)
            high = True

        if arr[i] < 0.5 and high:
            j = i
            while j < len(arr):
                if arr[j] > 0.5:
                    if (j - i) / sr > tolerence:
                        time_stamps.append(i / sr)
                        high = False
                    break 
                
                j += 1
            
            bool_arr += [high] * (j - i)
            i = j
        else:
            bool_arr += [high]
            i += 1
    
    print(time_stamps)
    return np.array(bool_arr)

def get_bool_arr(arr, threshold, window):
    bool_arr = list()
    i = 0
    while i < (len(arr) - window):
        tmp_arr = arr[i:i+window]
        if np.mean(tmp_arr) > threshold:
            bool_arr.extend(window * [True])
        else:
            bool_arr.extend(window * [False])
        i += window
    
    return np.array(bool_arr)

def smooth_signal(np_arr, window=5, passes=10):
    for i in range(passes):
        weights = np.ones(window) / window
        np_arr = np.convolve(np_arr, weights, "same")

    return np_arr

def write_to_audio_file(out_path, signal, sample_rate):
    librosa.output.write_wav(out_path, signal, sample_rate)

# Given an np_array of signal, will split into arrays into true sections.
def split_np_array(signal, bool_arr):
    indices = np.nonzero(bool_arr[1:] != bool_arr[:-1])[0] + 1
    splits = np.split(signal, indices)
    splits = splits[0::2] if bool_arr[0] else splits[1::2]
    return splits

def plot(*data):
    for i, values in enumerate(data):
        plt.subplot(len(data), 1, i+1)
        plt.plot(values)
    plt.show()

def getAudioText(filepath):
    try:
        files = ['clip0.wav', 'clip1.wav']
        for file in files:
            with open(join(dirname(__file__), './out', file),
                        'rb') as audio_file:
                speech_recognition_results = speech_to_text.recognize(
                    audio=audio_file,
                    content_type='audio/wav',
                    timestamps=True,
                ).get_result()
            print(json.dumps(speech_recognition_results, indent=2)) 
    except watson.WatsonApiException as ex:
        print("Method failed with status code " + str(ex.code) + ": " + ex.message)

def find_shortest_pause(signal):
    '''
    This finds the shortest pause.
    This is used to find the where to cut up the audio
    '''
    longest = len(signal)
    i = 0
    j = 0
    while i < len(signal):

        i += 1

if __name__ == "__main__":
    main()