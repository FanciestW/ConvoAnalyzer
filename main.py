import librosa
import numpy as np

def main():
    audio_file = "audio/dailylife002.wav"
    signal,sample_rate = librosa.load(audio_file, sr=None, mono=True)

    signal = np.array(signal)

    smoothed_signal = abs(signal)

    write_to_audio_file("out/before_smooth.wav", smoothed_signal, sample_rate)

    smoothed_signal = smooth_signal(smoothed_signal, int(sample_rate/500), 500)

    write_to_audio_file("out/after_smooth.wav", smoothed_signal, sample_rate)

    threshold = 0.01
    silence_window = int(sample_rate / 500)

    bool_signal = get_bool_arr(smoothed_signal, threshold, silence_window)
    bool_signal = smooth_signal(bool_signal, int(sample_rate/500), 100)
    write_to_audio_file("out/bool.wav", bool_signal.astype(float), sample_rate)

    cut_audio(bool_signal)

def cut_audio(arr, tolerence=0.2, sr=44100):
    time_stamps = list()
    
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
            
            i = j
        else:
            i += 1
    
    print(time_stamps)


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

# Given an array with a bool array, will split into arrays where true.
def split_np_array(a, b):
    indices = np.nonzero(b[1:] != b[:-1])[0] + 1
    c = np.split(a, indices)
    c = c[0::2] if b[0] else c[1::2]
    return c

if __name__ == "__main__":
    main()