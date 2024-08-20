import numpy as np
import matplotlib.pyplot as plt

# Given sequence
y = np.array([178, 261, 350, 524, 685, 949])

# Number of points in the sequence
n = len(y)

# Apply Fourier Transform
y_fft = np.fft.fft(y)

# Get the frequencies corresponding to the FFT coefficients
frequencies = np.fft.fftfreq(n)

# Sort the frequencies by magnitude
indices = np.argsort(np.abs(frequencies))

# Reconstruct the signal using the most significant frequencies
reconstructed_signal = np.zeros_like(y, dtype=complex)
for i in indices[:2]:  # Keep only the first two frequencies for simplicity
    reconstructed_signal += y_fft[i] * np.exp(2j * np.pi * frequencies[i] * np.arange(n))

# Predict the next value as the next point in the reconstructed signal
next_value_no_scaling = reconstructed_signal[-1].real
print(f"The predicted next value without scaling is approximately: {next_value_no_scaling:.0f}")