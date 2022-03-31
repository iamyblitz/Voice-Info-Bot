#функция превращает аудиозаписи в картинки
# -*- coding: utf-8 -*-
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pylab




def generate_spectogram(voice):

    y, sr = librosa.load(voice, mono=True, duration=5)

    fig, ax = plt.subplots()

    # trim silent edges

    whale_song, _ = librosa.effects.trim(y)

    librosa.display.waveplot(whale_song, sr=sr);

    n_fft = 2048

    D = np.abs(librosa.stft(whale_song[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))

    hop_length = 256

    D = np.abs(librosa.stft(whale_song, n_fft=n_fft, hop_length=hop_length))

    librosa.display.specshow(D, sr=sr, y_axis='linear');

    DB = librosa.amplitude_to_db(D, ref=np.max)

    librosa.display.specshow(DB, sr=sr, hop_length=hop_length, y_axis='log');

    pylab.axis('off')

    plt.gca().set_axis_off()

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    plt.margins(0, 0)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    voice = voice.split('/')[-1]

    fig.savefig('C:/Users/pic.png')
    path = 'C:/Users/pic.png'
    plt.clf()
    return path






