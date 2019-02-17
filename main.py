import librosa
import numpy as np
from scipy.fftpack import fft

def main():
    audio_file = "audio/dailylife002.wav"
    signal,sample_rate = librosa.load(audio_file, sr=None, mono=True)

    signal = np.array(signal)

    smoothed_signal = abs(signal)

    write_to_audio_file("out/before_smooth.wav", smoothed_signal, sample_rate)

    smoothed_signal = smooth_signal(smoothed_signal, int(sample_rate/250), 10)

    write_to_audio_file("out/after_smooth.wav", smoothed_signal, sample_rate)

    threshold = 0.015
    silence_window = int(sample_rate / 500)

    bool_signal = get_bool_arr(smoothed_signal, threshold, silence_window)
    bool_signal = smooth_signal(bool_signal, int(sample_rate/500), 100)
    bool_signal = cut_audio(bool_signal, 0.125)
    write_to_audio_file("out/bool.wav", bool_signal.astype(float), sample_rate)

    audio_clips = split_np_array(signal, bool_signal)
    gender_arr = []
    for i, clip in enumerate(audio_clips):
        write_to_audio_file(f"out/clip{i}.wav", clip, sample_rate)
        get_gender(clip)

def get_gender(signal, sr=44100):
    fourier = np.fft.rfft(signal)
    fourier_mag = np.abs(fourier)
    freq = np.fft.rfftfreq(signal.size, d=1./sr)
    max_freq = freq[np.argmax(fourier_mag)]
    # print(max_freq)
    if max_freq < 180:
        print("male")
    else:
        print("female")

def get_gender2(signal, sr=44100):
    fft_out = fft(signal)
    combined = fft(signal).ravel()

    meanfunfreeq = sum(combined)/combined.size
    if meanfunfreeq<0.14:
        print("male")
    else:
        print("female")

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

if __name__ == "__main__":
    main()