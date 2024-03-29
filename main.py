"""
Name: ConvoAnalyzer.py
Description: A Python3 program that transcribes male/female conversations
Author: William Lin
"""
import math
from os.path import join, dirname
import os
import sys
import argparse
import shutil
import numpy as np
import librosa
from matplotlib import pyplot as plt

from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud import WatsonApiException

MALE_FREQ_START = 65
MALE_FREQ_END = 185
FEMALE_FREQ_START = 165
FEMALE_FREQ_END = 255

DEBUG_HELP_TXT = "Flag to delete program output directory."
DIR_HELP_TXT = "Output directory for program output data. The default is ./out/"
INPUT_HELP_TXT = "Specifies the input .WAV file to be transcribed."
OUTPUT_HELP_TXT = ("Specifies the output file to write transcript to. "
                   "If none is specified, it will print to console.")

speech_to_text = SpeechToTextV1(
    iam_apikey='65qVRIfF-CQYgk268h9NJERzwiY-1xfRY6WCmpU9iF2L',
    url='https://stream.watsonplatform.net/speech-to-text/api'
)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help=INPUT_HELP_TXT)
parser.add_argument("-o", "--output", help=OUTPUT_HELP_TXT)
parser.add_argument("-d", "--dir", help=DIR_HELP_TXT)
parser.add_argument("--debug", action="store_true", help=DEBUG_HELP_TXT)

def main():
    DEBUG_MODE = False
    AUDIO_FILE = "audio/custom01.wav"
    OUT_DIR = "./out/"

    # Parse passed arguments.
    args = parser.parse_args()
    if args.input:
        AUDIO_FILE = args.input
    if args.output:
        sys.stdout = open(args.output, 'w')
    if args.dir:
        OUT_DIR = args.dir + '/' if not args.dir.endswith('/') else args.dir
    if args.debug:
        DEBUG_MODE = True

    signal, sample_rate = librosa.load(AUDIO_FILE, sr=None, mono=True)
    smoothed_signal = abs(signal)

    if not os.path.exists(os.path.dirname(OUT_DIR)):
        try:
            os.makedirs(os.path.dirname(OUT_DIR))
        except Exception as ex: # Guard against race condition
            print(ex)

    if DEBUG_MODE: write_to_audio_file(f"{OUT_DIR}before_smooth.wav", smoothed_signal, sample_rate)
    smoothed_signal = smooth_signal(smoothed_signal, 10, 10)
    if DEBUG_MODE: write_to_audio_file(f"{OUT_DIR}after_smooth.wav", smoothed_signal, sample_rate)

    threshold = 0.015
    silence_window = int(sample_rate / 150)

    bool_signal = get_silence_bool(smoothed_signal, threshold, silence_window)
    bool_signal = smooth_signal(bool_signal, int(sample_rate/150), 50)

    if DEBUG_MODE: write_to_audio_file(f"{OUT_DIR}before_bool.wav", bool_signal.astype(float), sample_rate)
    time_stamps = get_cut_times(bool_signal, tolerence=0.15, sr=sample_rate)
    bool_signal = get_cut_bool_arr(signal, time_stamps, sr=sample_rate, padding=int(sample_rate/10))
    if DEBUG_MODE: write_to_audio_file(f"{OUT_DIR}after_bool.wav", bool_signal.astype(float), sample_rate)

    audio_clips = split_np_array(signal, bool_signal, padding=int(sample_rate/10))
    gender_arr = []
    for i, clip in enumerate(audio_clips):
        write_to_audio_file(f"{OUT_DIR}clip{i}.wav", clip, sample_rate)
        gender_arr.append(get_gender(smooth_signal(clip, window=10, passes=10)))

    results = combine_gender_audio(audio_clips, time_stamps, gender_arr)
    final_clips = results['clips']
    time_stamps = results['timestamps']
    gender_arr = results['genders']
    for i, clip in enumerate(final_clips):
        write_to_audio_file(f"{OUT_DIR}final_clip{i}.wav", clip, sample_rate)

    transcripts = []
    for i in range(len(final_clips)):
        transcripts.append(getAudioText(join(dirname(__file__), OUT_DIR, f'final_clip{i}.wav')))

    for time, gender, text in zip(time_stamps, gender_arr, transcripts):
        print("%8.3fs %6s - %s" %(time, gender, text))

    if not DEBUG_MODE:
        shutil.rmtree(OUT_DIR)

def get_gender(signal, sr=44100):
    '''
    Given a clip of audio named signal, return the potential gender
    '''
    signal = smooth_signal(signal, 50, 10)
    male_points = 0
    female_points = 0
    small_signals = np.array_split(signal, 4)
    for sig in small_signals:
        fourier = np.fft.rfft(sig)
        fourier_mag = np.abs(fourier)
        fourier_mag = smooth_signal(fourier_mag, 10, 10)
        freq = np.fft.rfftfreq(sig.size, d=1./sr)
        start_male_range = np.argmax(freq >= MALE_FREQ_START)
        end_male_range = np.argmax(freq >= MALE_FREQ_END)
        start_female_range = np.argmax(freq >= FEMALE_FREQ_START)
        end_female_range = np.argmax(freq >= FEMALE_FREQ_END)

        male_avg = np.mean(fourier_mag[start_male_range:end_male_range])
        female_avg = np.mean(fourier_mag[start_female_range:end_female_range])
        max_freq = freq[np.argmax(fourier_mag)]
        # print(male_avg, female_avg, np.argmax(fourier_mag), max_freq)
        if male_avg >= female_avg:
            male_points += 1
        elif male_avg < female_avg:
            female_points += 1
        if MALE_FREQ_START <= max_freq <= MALE_FREQ_END:
            male_points += 1
        if FEMALE_FREQ_START <= max_freq <= FEMALE_FREQ_END:
            female_points += 1

    if male_points > female_points:
        # print("male")
        return "male"
    else:
        # print("female")
        return "female"

def combine_gender_audio(signals_arr, timestamps, gender_arr):
    '''
    Combines audio clips that have the same gender back to back.
    This will also return timestamps of the new clips of audio
    '''
    if len(signals_arr) != len(gender_arr) or len(signals_arr) < 1 or len(gender_arr) < 1:
        return None, None
    elif math.ceil(len(timestamps) / 2) < len(gender_arr):
        return None, None
    timestamps = timestamps[::2]
    new_timestamps = []
    new_gender_arr = []
    all_signals = []
    clip = np.array([])
    tmp_gender = None
    j = 0
    for i, gender in enumerate(gender_arr):
        if tmp_gender == None:
            tmp_gender = gender
            new_gender_arr.append(gender)
        elif tmp_gender != gender:
            clip = np.concatenate(signals_arr[j:i]).ravel().tolist()
            new_timestamps.extend([timestamps[j]])
            new_gender_arr.append(gender)
            all_signals.append(np.array(clip))
            clip = np.array([])
            tmp_gender = gender
            j = i
        if i >= len(gender_arr)-1:
            new_timestamps.extend([timestamps[j]])

    clip = np.concatenate(signals_arr[j:]).ravel().tolist()
    all_signals.append(np.array(clip))
    return {'clips': all_signals, 'timestamps': new_timestamps, 'genders': new_gender_arr}

def get_cut_times(bool_arr, tolerence=0.2, sr=44100):
    '''
    Given a bool array, returns timestamps of chunks where the
    bool array is low(false) or a given amount of time (tolerence).
    '''
    time_stamps = list()
    high = False
    i = 0
    j = 0
    while i < len(bool_arr):
        # Start of new audio segment
        if bool_arr[i] >= 0.4 and not high:
            time_stamps.append(i / sr)
            high = True

        if bool_arr[i] < 0.4 and high:
            j = i
            while j < len(bool_arr):
                if bool_arr[j] > 0.5:
                    if (j - i) / sr > tolerence:
                        time_stamps.append(i / sr)
                        high = False
                    break
                j += 1
            i = j
        else:
            i += 1
    return time_stamps

def get_cut_bool_arr(signal, timestamps, sr=44100, padding=0):
    '''
    Given timestamps and the original signal array, a boolean array will be created
    that corresponds the start and stop of where to cut the audio into pieces.
    '''
    bool_arr = []
    timestamps = (np.array(timestamps) * sr).astype(int).tolist()
    for i, time in enumerate(timestamps):
        if i % 2 == 0:
            timestamps[i] -= padding
        else:
            timestamps[i] += padding

    last = 0
    for i, time in enumerate(timestamps):
        if i % 2 == 0:
            bool_arr += ((time - last) * [False])
            last = time
        else:
            bool_arr += ((time - last) * [True])
            last = time
    if len(timestamps) % 2 == 1:
        bool_arr += ((len(signal) - last) * [True])

    return np.array(bool_arr)

def get_silence_bool(arr, threshold, window):
    '''
    Returns a boolean array the maps silences that surpass a certain threshold.
    The boolean array has false for silence and true for audio activity.
    '''
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

# Given an numpy array, smooth the array using a window and number of passes.
def smooth_signal(np_arr, window=5, passes=10):
    while passes > 0:
        weights = np.ones(window) / window
        np_arr = np.convolve(np_arr, weights, "same")
        passes -= 1
    return np_arr

# Writes signal array to a output .wav file.
def write_to_audio_file(out_path, signal, sample_rate):
    librosa.output.write_wav(out_path, signal, sample_rate)

# Given an np_array of signal, will split into arrays into true sections.
def split_np_array(signal, bool_arr, padding=0, pad_val=0.):
    indices = np.nonzero(bool_arr[1:] != bool_arr[:-1])[0] + 1
    splits = np.split(signal, indices)
    splits = splits[0::2] if bool_arr[0] else splits[1::2]
    for i, clip in enumerate(splits):
        splits[i] = np.pad(clip, (padding, padding), 'constant', constant_values=(pad_val, pad_val))
    return splits

# Plots arrays onto a pyplot figure with graphs.
def plot(*data):
    for i, values in enumerate(data):
        plt.subplot(len(data), 1, i+1)
        plt.plot(values)
    plt.show()

# Uses IBM Watson's speech recognition to get speech to text results
def getAudioText(filepath):
    try:
        with open(filepath, 'rb') as AUDIO_FILE:
            speech_recognition_results = speech_to_text.recognize(
                audio=AUDIO_FILE,
                content_type='audio/wav',
                timestamps=True,
            ).get_result()
        # print(speech_recognition_results["results"][0]["alternatives"][0]["transcript"])
        return speech_recognition_results["results"][0]["alternatives"][0]["transcript"]
    except WatsonApiException as ex:
        print("Method failed with status code " + str(ex.code) + ": " + ex.message)

if __name__ == "__main__":
    main()
