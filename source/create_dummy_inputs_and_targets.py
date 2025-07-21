import numpy as np
import os

# Output folders
os.makedirs("data/raw", exist_ok=True)

sample_rate = 8000  # Hz
duration = 1.0      # seconds
t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)

# Frequencies
base_frequencies = [110, 220, 440, 880, 1760]

for i, freq in enumerate(base_frequencies):
    input_wave = 0.5 * np.sin(2 * np.pi * freq * t)
    target_wave = 0.5 * np.sin(2 * np.pi * (2 * freq) * t)

    # Save as TXT
    np.savetxt(f"data/raw/dummy_input_{i}.txt", input_wave)
    np.savetxt(f"data/raw/dummy_target_{i}.txt", target_wave)
