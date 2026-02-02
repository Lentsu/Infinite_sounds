#!/usr/bin/env python3
"""
Freeze audio effect using velvet noise or fft convolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, fftconvolve
from enum import Enum
import argparse
import soundfile as sf
import simpleaudio as sa

class Window(Enum):
    TRIANGULAR = 1
    PARZEN = 2
    NUTTAL = 3
    SINUSOIDAL = 4
    WELCH = 5


def window(n, N, w):
    """ Constructs a window function.
    
    Keyword arguments:
    n -- the input array
    N -- window length in samples
    w -- window type identifier (enum)
    """
    n = np.asarray(n, dtype=float)

    if w == Window.TRIANGULAR:
        # Triangular
        y = 1.0 - np.abs(2 * n/N - 1)
    elif w == Window.PARZEN:
        # Parzen
        n2 = n - N/2
        y = np.zeroes_like(n2)
        mask1 = np.abs(n2) < N / 4
        mask2 = ~mask1
        y[mask1] = 1 - 6*(2*n2[mask1] / N)**2 * (1 - 2*np.abs(n2[mask1]) / N)
        y[mask2] = 2*(1 - 2*np.abs(n2[mask2]) / N)**3
    elif w == Window.NUTTAL:
        # Nuttal
        a0 = 0.355768
        a1 = 0.487369
        a2 = 0.144232
        a3 = 0.012604
        y = (
            a0
            - a1 * np.cos(2*np.pi * n/(N - 1))
            + a2 * np.cos(4*np.pi * n/(N - 1))
            - a3 * np.cos(6*np.pi * n/(N - 1))
        )
    elif w == Window.SINUSOIDAL:
        # Sinusoidal
        y = np.sin(np.pi * n/(N - 1))
    else:
        # Welch
        y = 1 - ((n - (N - 1)/2) / ((N - 1)/2))**2

    return y


def plot_signals(x, n, xg, y, fs, plot_file=None):
    """Plots the velvet-noise freeze effect signals

    Keyword arguments:
    x  -- the input signal
    n  -- velvet noise signal
    xg -- granulated input signal
    y  -- the output signal
    fs -- sampling rate in Hz
    plot_file -- (optional) audio output filename
    """
    # Plot the time domain signals
    t_x = np.arange(len(x)) / fs
    t_xg = np.arange(len(xg)) / fs
    t_y = np.arange(len(y)) / fs

    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=False)

    axs[0].plot(t_x, x)
    axs[0].set_title("Input signal")

    axs[1].stem(t_y, n)
    axs[1].set_title("Velvet noise")

    axs[2].plot(t_xg, xg)
    axs[2].set_title("Granulated input")

    axs[3].plot(t_y, y)
    axs[3].set_title("Output signal")

    for a in axs:
        a.set_xlabel("Time [s]")
        a.set_ylabel("Amplitude")
        a.grid(True)

    plt.tight_layout()

    # Plot the input and output spectrograms
    fig_spec,axs_spec = plt.subplots(2, 1, figsize=(12,6), sharex=False)

    axs_spec[0].specgram(x, Fs=fs, scale="dB", cmap="grey")
    axs_spec[0].set_title("Input signal")
    axs_spec[0].set_ylabel("Freq. [Hz]")

    axs_spec[1].specgram(y, Fs=fs, scale="dB", cmap="grey")
    axs_spec[1].set_title("Output signal")
    axs_spec[1].set_ylabel("Freq. [Hz]")

    plt.tight_layout()

    # Show or save the plots
    if plot_file is None:
        plt.show(block=False)
    else:
        fig.savefig(plot_file.replace(".png", "_signals.png"))
        fig_spec.savefig(plot_file.replace(".png", "_spectrogram.png"))
        plt.close(fig)


def freeze(x, fs, ti, to, d, w, plot_file=None):
    """Creates the freeze effect using velvet-noise convolution
    
    Keyword arguments:
    x  -- the input signal
    fs -- sampling rate in Hz
    ti -- input duration in seconds
    to -- output duration in seconds
    d  -- average grain density
    w  -- window type identifier (enum)
    """

    x = np.asarray(x, dtype=float)

    N = int(round(ti * fs))
    No = int(round(to * fs))

    # Generate velvet-noise
    nx = np.random.rand(No)
    low = d / (2 * N)
    high = 1 - low
    n = np.zeros(No)
    n[nx > high] = 1
    n[nx < low] = -1

    # Estimate input level
    tau = 0.06514417228548776
    mA1 = np.exp(-1 / (fs * tau))
    B0 = 1 - mA1

    # Filter
    lx = lfilter(
        [B0],
        [1, -mA1],
        20 * np.log10(np.maximum(np.abs(x), 1e-6)),
        zi=[-120]
    )[0][-1]

    # Granulate input
    xg = x[-N:] * window(np.arange(N), N, w)

    # Convolve
    y = fftconvolve(xg, n, mode="full")[:No]

    # Compensate output level
    ly = lfilter(
        [B0],
        [1, -mA1],
        20 * np.log10(np.maximum(np.abs(y), 1e-6)),
        zi=[20 * np.log10(d / ti)]
    )[0]

    y *= 10**(np.minimum(lx - ly, 6) / 20)
    #y *= 10**((lx - ly) / 20)

    # Plot the signals
    plot_signals(x, n, xg, y, fs, plot_file)

    return y


def play_audio(x,fs):
    """Plays the audio signal x with sample rate fs"""
    # Normalize the audio
    max_val = np.max(np.abs(x))
    if max_val > 1:
        x = x / max_val
    # Convert to 16-bit PCM
    y_PCM16 = np.int16(x * 32767)
    # Play
    play_obj = sa.play_buffer(
        y_PCM16,
        num_channels=1,
        bytes_per_sample=2,
        sample_rate=fs
    )

def main():
    """ CLI entry point for the script

    Command-line arguments:
    input    -- path to input audio file
    output   -- path to output audio file
    duration -- output duration in seconds
    density  -- average grain density
    """
    # Parse the CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("duration", type=float)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--window", type=int, default=Window.NUTTAL)
    parser.add_argument("--grain", type=float, default=None)
    parser.add_argument("--density", type=float, default=40.0)
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    # Assign the arguments to variables
    to = args.duration
    d = args.density
    w = args.window

    # Extract the samples and sampling rate from the input file
    x,fs = sf.read(args.input)

    # Turn the sound to mono
    if x.ndim > 1:
        x = x.mean(axis=1)
    
    # Calculate the grain size (if not given)
    if args.grain is None:
        ti = len(x) / fs
    else:
        ti = args.grain

    # Apply the freeze effect
    y = freeze(x, fs, ti, to, d, w, args.plot)

    # Write the frozen sound to the output file (if given)
    if args.output is not None:
        sf.write(args.output, y, fs)
    # Else, play the audio
    else:
        play_audio(y,fs)

    # Wait input before ending the script (closing plots)
    input("[Press Enter]")

if __name__ == "__main__":
    main()
