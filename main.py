import librosa
import numpy as np

def main():
    audio_file = "audio/dailylife002.wav"
    signal,sample_rate = librosa.load(audio_file, sr=None, mono=True)

    signal = np.array(signal)
    positive_audio_bool = (signal > 0.1) | (signal < -0.1)
    positive_audio = signal[positive_audio_bool]

    window_size = 4
    weights = np.ones(window_size) / window_size
    avg_signal = np.convolve(signal, weights, "same")

    true_positive_audio_bool = (avg_signal > 0.1) | (avg_signal < -0.1)
    true_positive_audio = avg_signal[true_positive_audio_bool]


    print("Sample Rate:", sample_rate)
    print(len(positive_audio)/sample_rate)
    print(len(true_positive_audio)/sample_rate)

# Given an array with a bool array, will split into arrays where true.
def split_np_array(a, b):
    indices = np.nonzero(b[1:] != b[:-1])[0] + 1
    c = np.split(a, indices)
    c = c[0::2] if b[0] else c[1::2]
    return c

if __name__ == "__main__":
    main()