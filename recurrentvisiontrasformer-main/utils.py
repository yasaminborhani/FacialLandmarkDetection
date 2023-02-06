from scipy import signal

def spectrogram(sig, fs):
    f, t, Sxx = signal.spectrogram(sig, fs)
    return f, t, Sxx
