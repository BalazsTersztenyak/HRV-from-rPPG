import numpy as np
import matplotlib.pyplot as plt
import hrv

rr_file = "data/BSP_Projekt/2025-11-28 16-27-20_base.txt"
csv_path = "output/BSP_Projekt/base/yuv.csv"

# load RR intervals from file (ms)
with open(rr_file) as f:
    rr_ms = np.array([float(line.strip()) for line in f if line.strip()])

# convert to seconds
rr = rr_ms / 1000.0

fs = 500

def synthetic_beat(length):
    t = np.linspace(0, 1, length)
    p = 0.1 * np.sin(2 * np.pi * 5 * t)
    qrs = -np.exp(-((t - 0.3)**2) / 0.0005) + 3 * np.exp(-((t - 0.32)**2) / 0.0001) - np.exp(-((t - 0.34)**2) / 0.0005)
    t_wave = 0.2 * np.exp(-((t - 0.7)**2) / 0.01)
    return p + qrs + t_wave

ecg = np.array([])

for interval in rr:
    samples = int(interval * fs)
    beat = synthetic_beat(samples)
    ecg = np.concatenate([ecg, beat])

time = np.arange(len(ecg))/fs

yuv = np.loadtxt(csv_path, delimiter=',', skiprows=1)

yuv_ma = hrv.magnify_colour_ma(
        np.array(yuv, dtype=np.float64),
        delta=1,
        n_bg_ma=90,
        n_smooth_ma=6
        )

yuv_ma_filtered = hrv.filter_signal(yuv_ma, 29.75)
yuv_peaks = hrv.detect_beats(yuv_ma_filtered[..., 0], 29.75)/29.75


plt.figure(figsize=(12,4))
# plt.plot(time, ecg)
plt.plot(np.arange(len(yuv_ma_filtered))/29.75, yuv_ma_filtered)
for p in yuv_peaks:
    plt.axvline(p, color='blue', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("ECG-like Signal Generated from RR Intervals")
plt.tight_layout()
plt.show()
