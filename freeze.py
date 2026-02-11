#!/usr/bin/env python3

"""
Freeze audio effect using velvet noise convolution or FFT method.

Usage:
python freeze.py <input.wav> <new duration [s]> (--output <output.wav>)
(--method <freezing method>) (--window <window type id>) (--grain <grain-size
[s]>) (--density <grain-density>) (--plot <plot_filepath.png>)
"""

import argparse
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, fftconvolve
import soundfile as sf
import simpleaudio as sa

# Suppress divide by 0 warnings
np.seterr(divide='ignore')


class Method(Enum):
    """Enumerator for implemented signal freezing methods"""
    VELVET_NOISE_CONVOLUTION_NO_COMPENSATION = 1
    VELVET_NOISE_CONVOLUTION = 2
    RANDOM_PHASE_VOCODER = 3


class Window(Enum):
    """Enumerator for implemented window types"""
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
        y = 1.0 - np.abs(2 * n / N - 1)
    elif w == Window.PARZEN:
        # Parzen
        n2 = n - N / 2
        y = np.zeroes_like(n2)
        mask1 = np.abs(n2) < N / 4
        mask2 = ~mask1
        y[mask1] = 1 - 6 * (2 * n2[mask1] / N)**2 * \
            (1 - 2 * np.abs(n2[mask1]) / N)
        y[mask2] = 2 * (1 - 2 * np.abs(n2[mask2]) / N)**3
    elif w == Window.NUTTAL:
        # Nuttal
        a0 = 0.355768
        a1 = 0.487369
        a2 = 0.144232
        a3 = 0.012604
        y = (
            a0
            - a1 * np.cos(2 * np.pi * n / (N - 1))
            + a2 * np.cos(4 * np.pi * n / (N - 1))
            - a3 * np.cos(6 * np.pi * n / (N - 1))
        )
    elif w == Window.SINUSOIDAL:
        # Sinusoidal
        y = np.sin(np.pi * n / (N - 1))
    else:
        # Welch
        y = 1 - ((n - (N - 1) / 2) / ((N - 1) / 2))**2

    return y


def plot_signals(x, xg, y, fs, m, plot_file=None, n=None):
    """Plots the freeze effect signals.

    Keyword arguments:
    x  -- the input signal
    xg -- frozen frame (input grain)
    y  -- the output signal
    fs -- sampling rate in Hz
    m  -- the freezing effect methods
    plot_file -- (optional) audio output filename
    n  -- velvet noise signal if Method.VELVET_NOISE_CONVOLUTION
    """
    # Plot the time domain signals
    t_x = np.arange(len(x)) / fs
    t_xg = np.arange(len(xg)) / fs
    t_y = np.arange(len(y)) / fs

    # Plot velvet noise only for velvet noise convolution
    if m == Method.VELVET_NOISE_CONVOLUTION:
        fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=False)
    else:
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    axs[0].plot(t_x, x)
    axs[0].set_title("Input signal")

    axs[1].plot(t_xg, xg)
    axs[1].set_title("Granulated input")

    axs[2].plot(t_y, y)
    axs[2].set_title("Output signal")

    # Plot velvet noise only for velvet noise convolution
    if m == Method.VELVET_NOISE_CONVOLUTION:
        axs[3].stem(t_y, n)
        axs[3].set_title("Velvet noise")

    for a in axs:
        a.set_xlabel("Time [s]")
        a.set_ylabel("Amplitude")
        a.grid(True)

    plt.tight_layout()

    # Plot the input and output spectrograms
    fig_spec, axs_spec = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    axs_spec[0].specgram(x, Fs=fs, scale="dB", cmap="grey")
    axs_spec[0].set_title("Input spectrogram")
    axs_spec[0].set_xlabel("Time [s]")
    axs_spec[0].set_ylabel("Freq. [Hz]")

    axs_spec[1].specgram(y, Fs=fs, scale="dB", cmap="grey")
    axs_spec[1].set_title("Output spectrogram")
    axs_spec[1].set_xlabel("Time [s]")
    axs_spec[1].set_ylabel("Freq. [Hz]")

    plt.tight_layout()
    
    # Plot the input and output spectrums 
    fig_mag, axs_mag = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    # Input spectrum
    Xf = np.fft.rfft(x)
    fx = np.fft.rfftfreq(len(x), 1/fs)
    XM = 20 * np.log10(np.abs(Xf) + 1e-12)
    axs_mag[0].plot(fx, XM)
    axs_mag[0].set_title("Input spectrum")
    axs_mag[0].set_xlabel("Freq. [Hz]")
    axs_mag[0].set_ylabel("Mag. [dB]")
    axs_mag[0].grid(True)

    # Output spectrum
    Yf = np.fft.rfft(y)
    fy = np.fft.rfftfreq(len(y), 1/fs)
    YM = 20 * np.log10(np.abs(Yf) + 1e-12)
    axs_mag[1].plot(fy, YM)
    axs_mag[1].set_title("Output spectrum")
    axs_mag[1].set_xlabel("Freq. [Hz]")
    axs_mag[1].set_ylabel("Mag. [dB]")
    axs_mag[1].grid(True)

    plt.tight_layout()

    # Show or save the plots
    if plot_file is None:
        plt.show(block=False)
    else:
        fig.savefig(plot_file.replace(".png", "_signals.png"))
        fig_spec.savefig(plot_file.replace(".png", "_spectrogram.png"))
        fig_mag.savefig(plot_file.replace(".png", "_spectrum.png"))
        plt.close(fig)


def freeze_velvet(x, fs, ti, to, d, w, plot_file=None, compensate=True):
    """Creates the freeze effect using velvet-noise convolution

    Keyword arguments:
    x  -- the input signal
    fs -- sampling rate in Hz
    ti -- input duration in seconds
    to -- output duration in seconds
    d  -- average grain density
    w  -- window type identifier (enum)
    plot_file -- (optional) audio output filename
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

    # Granulate input
    xg = x[-N:] * window(np.arange(N), N, w)

    # Convolve
    y = fftconvolve(xg, n, mode="full")[:No]

    # If compensation is enabled
    if compensate:
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
        # Compensate output level
        ly = lfilter(
            [B0],
            [1, -mA1],
            20 * np.log10(np.maximum(np.abs(y), 1e-6)),
            zi=[20 * np.log10(d / ti)]
        )[0]
        # Compensated output is more smooth but lower volume
        y *= 10**(np.minimum(lx - ly, 6) / 20)
    else:
        # Just Normalized output is high volume but isn't smooth
        y /= np.max(np.abs(y)) + 1e-12    # 1e-12 to avoid divide by 0

    # Plot the signals
    plot_signals(x, xg, y, fs, Method.VELVET_NOISE_CONVOLUTION, plot_file, n)

    return y


def freeze_fft(x, fs, ti, to, d, w, plot_file=None):
    """Creates the freeze effect using random phase vocoder (fft-based technique) 

    Keyword arguments:
    x  -- the input signal
    fs -- sampling rate in Hz
    ti -- input duration in seconds
    to -- output duration in seconds
    d  -- average grain density
    w  -- window type identifier (enum)
    plot_file -- (optional) audio output filename
    """
    x = np.asarray(x, dtype=float)

    # Grain length, output length
    N = int(round(ti * fs))
    No = int(round(to * fs))

    # Create the window
    win = window(np.arange(N), N, w)

    # Input grain (windowed)
    xg = x[-N:] * win

    # Zero-pad the input grain
    xg_pad = np.zeros(No)
    xg_pad[:N] = xg

    # FFT of input grain 
    Xf = np.fft.fft(xg_pad)
    Rx = np.abs(Xf)

    # Random phase values
    theta_r = np.random.uniform(-np.pi, np.pi, len(Rx))

    # Rx * exp(j*\theta_r)
    Yf = Rx * np.exp(1j * theta_r)

    # IFFT -> output signal
    y = np.real(np.fft.ifft(Yf))

    # Normalize
    y /= np.max(np.abs(y)) + 1e-12    # 1e-12 to avoid divide by 0

    # Plot the signals
    plot_signals(x, xg_pad, y, fs, Method.RANDOM_PHASE_VOCODER, plot_file)
    
    return y


def play_audio(x, fs):
    """Plays the audio signal x with sample rate fs"""
    # Normalize the audio
    max_val = np.max(np.abs(x))
    if max_val > 1:
        x = x / max_val
    # Convert to 16-bit PCM
    y_PCM16 = np.int16(x * 32767)
    # Play
    sa.play_buffer(
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
    method   -- method for the freeze effect
    window   -- window type identifier
    grain    -- the grain/window size in seconds
    density  -- average grain density
    plot     -- path to output plot file(s)
    """
    usage = (
        "python freeze.py <input.wav> <new duration [s]>\n"
        "    (--output <output.wav>)\n"
        "    (--method <freezing method>)\n"
        "    (--window <window type id>)\n"
        "    (--grain <grain-size [s]>)\n"
        "    (--density <grain-density>)\n"
        "    (--plot <plot_filepath.png>)"
    )
    description = "Freeze audio effect using velvet noise convolution or FFT method"

    # Parse the CLI arguments
    parser = argparse.ArgumentParser(usage, description)
    parser.add_argument("input", type=str,
                        help="path to input audio file")
    parser.add_argument("duration", type=float,
                        help="output duration in seconds")
    parser.add_argument("--method", type=int, default=Method.VELVET_NOISE_CONVOLUTION,
                        help="method for the freeze effect")
    parser.add_argument("--output", type=str, default=None,
                        help="path to output audio file")
    parser.add_argument("--window", type=int, default=Window.NUTTAL,
                        help="window type identifier (enum)")
    parser.add_argument("--grain", type=float, default=None,
                        help="grain/window size in seconds")
    parser.add_argument("--density", type=float, default=40.0,
                        help="average grain density")
    parser.add_argument("--plot", type=str, default=None,
                        help="path to output plot file")
    args = parser.parse_args()

    # Assign the arguments to variables
    to = args.duration
    d = args.density
    
    # Get the window type from args
    try:
        w = Window(args.window)
    except:
        parser.error(f"Invalid window type '{args.window}'.")
    
    # Get the freezing method from args
    try:
        m = Method(args.method)
    except:
        parser.error(f"Invalid freezing method '{args.method}'.")
    
    # Extract the samples and sampling rate from the input file
    x, fs = sf.read(args.input)

    # Turn the sound to mono
    if x.ndim > 1:
        x = x.mean(axis=1)

    # Calculate the grain size (if not given)
    if args.grain is None:
        ti = len(x) / fs
    else:
        ti = args.grain

    # Apply the freeze effect
    if m == Method.VELVET_NOISE_CONVOLUTION_NO_COMPENSATION:
        y = freeze_velvet(x, fs, ti, to, d, w, args.plot, compensate=False)
    elif m == Method.VELVET_NOISE_CONVOLUTION:
        y = freeze_velvet(x, fs, ti, to, d, w, args.plot)
    elif m == Method.RANDOM_PHASE_VOCODER:
        y = freeze_fft(x, fs, ti, to, d, w, args.plot)
    else:
        # Invalid arguments
        parser.error(
            f"Invalid freezing method '{args.method}'. "
        )

    # Write the frozen sound to the output file (if given)
    if args.output is not None:
        sf.write(args.output, y, fs)
    # Else, play the audio
    else:
        play_audio(y, fs)
        # Wait input before ending the script (closing plots)
        input("[Press Enter]")


if __name__ == "__main__":
    main()
