# region Imports
import datetime
from multiprocessing import process
from turtle import color
from venv import create
import cv2
from cv2 import CascadeClassifier
from cv2.data import haarcascades
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import math
from scipy.signal import find_peaks
import glob
# endregion

def main():
    txt_files = glob.glob("data/BSP_Projekt/**/*.txt", recursive=True)
    video_files = glob.glob("data/BSP_Projekt/**/*.mp4", recursive=True)

    data = {}
    for file_path in txt_files:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            values = [float(line.strip())/1000 for line in lines if line.strip()]
            data[file_path] = values
    
    with open('hrv_results.csv', 'w') as result_file:
        for file_path, rr_intervals in data.items():
            output_path = f"output/{file_path.split('/')[-1].split('.')[0]}"
            os.makedirs(output_path, exist_ok=True)
            rr_intervals = drop_outliers(np.array(rr_intervals))*1000
            sdnn_values, rmssd_values = sliding_window_hrv(rr_intervals)
            result_file.write(f"{file_path},{sdnn_values},{rmssd_values}\n")
            create_plot(None,
                        sdnn_values,
                        'SDNN',
                        'File',
                        'SDNN (ms)',
                        ['SDNN'],
                        f'{output_path}/sdnn.png')
            create_plot(None,
                        rmssd_values,
                        'RMSSD',
                        'File',
                        'RMSSD (ms)',
                        ['RMSSD'],
                        f'{output_path}/rmssd.png')

    for path in video_files:
        process_video(path)

def process_video(path):
    output_path = f"output/{path.split('/')[-1].split('.')[0]}"
    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(f'{output_path}/rgb.csv') and os.path.exists(f'{output_path}/yuv.csv'):
        print(f"Skipping {path}, already processed.")
        rgb = np.loadtxt(f'{output_path}/rgb.csv', delimiter=',', skiprows=1)
        yuv = np.loadtxt(f'{output_path}/yuv.csv', delimiter=',', skiprows=1)
        fps = get_fps(path)
    else:
        rgb, yuv, fps = extract_avg_rgb(path)

    save_csv(f'{output_path}/rgb.csv', rgb, header=['Blue', 'Green', 'Red'])
    save_csv(f'{output_path}/yuv.csv', yuv, header=['Y', 'U', 'V'])

    rgb_ma = magnify_colour_ma(
        np.array(rgb, dtype=np.float64),
        delta=1,
        n_bg_ma=90,
        n_smooth_ma=6
        )
    yuv_ma = magnify_colour_ma(
        np.array(yuv, dtype=np.float64),
        delta=1,
        n_bg_ma=90,
        n_smooth_ma=6
        )

    rgb_ma_filtered = filter_signal(rgb_ma, fps)
    rgb_peaks = detect_beats(rgb_ma_filtered[..., 1], fps)
    rgb_rr_intervals = compute_rr(rgb_peaks, fps)
    yuv_ma_filtered = filter_signal(yuv_ma, fps)
    yuv_peaks = detect_beats(yuv_ma_filtered[..., 0], fps)
    yuv_rr_intervals = compute_rr(yuv_peaks, fps)
    
    sdnn, rmssd = compute_hrv_metrics(rgb_rr_intervals)
    print(f"RGB: \nSDNN: {sdnn:.2f} ms, RMSSD: {rmssd:.2f} ms")
    sdnn, rmssd = compute_hrv_metrics(yuv_rr_intervals)
    print(f"YUV: \nSDNN: {sdnn:.2f} ms, RMSSD: {rmssd:.2f} ms")

    sdnn_values, rmssd_values = sliding_window_hrv(rgb_rr_intervals)
    create_plot(None,
                sdnn_values,
                'Sliding Window SDNN',
                'Window',
                'SDNN (ms)',
                ['SDNN'],
                f'{output_path}/sliding_window_sdnn.png')
    create_plot(None,
                rmssd_values,
                'Sliding Window RMSSD',
                'Window',
                'RMSSD (ms)',
                ['RMSSD'],
                f'{output_path}/sliding_window_rmssd.png')
    
    sdnn_values, rmssd_values = sliding_window_hrv(yuv_rr_intervals)
    create_plot(None,
                sdnn_values,
                'Sliding Window SDNN YUV',
                'Window',
                'SDNN (ms)',
                ['SDNN'],
                f'{output_path}/sliding_window_sdnn_yuv.png')
    create_plot(None,
                rmssd_values,
                'Sliding Window RMSSD YUV',
                'Window',
                'RMSSD (ms)',
                ['RMSSD'],
                f'{output_path}/sliding_window_rmssd_yuv.png')

    create_plot(None, 
                rgb,
                'RGB', 
                'Frame', 
                'Intensity', 
                ['Blue', 'Green', 'Red'], 
                f'{output_path}/rgb.png')
    create_plot(None, 
                yuv,
                'YUV', 
                'Frame', 
                'Intensity', 
                ['Y', 'U', 'V'], 
                f'{output_path}/yuv.png')
    
    create_plot(None, 
                rgb_ma_filtered[..., 1],
                'Filtered Magnified Green', 
                'Frame', 
                'Intensity', 
                ['Green_Filtered_MA'], 
                f'{output_path}/g_ma_filtered.png')
    create_plot(None, 
                yuv_ma_filtered[..., 0],
                'Filtered Magnified Y', 
                'Frame', 
                'Intensity', 
                ['Y_Filtered_MA'], 
                f'{output_path}/y_ma_filtered.png')

    create_plot(None, 
                rgb_ma,
                'Magnified Colour MA', 
                'Frame', 
                'Intensity', 
                ['Blue_MA', 'Green_MA', 'Red_MA'], 
                f'{output_path}/rgb_ma.png')
    create_plot(None, 
                yuv_ma,
                'Magnified Colour MA YUV', 
                'Frame', 
                'Intensity', 
                ['Y_MA', 'U_MA', 'V_MA'], 
                f'{output_path}/yuv_ma.png')
    
    create_plot(None, 
                rgb_ma_filtered,
                'Filtered Magnified Colour MA', 
                'Frame', 
                'Intensity', 
                ['Blue_Filtered_MA', 'Green_Filtered_MA', 'Red_Filtered_MA'], 
                f'{output_path}/rgb_ma_filtered.png')
    create_plot(None, 
                yuv_ma_filtered,
                'Filtered Magnified Colour MA YUV', 
                'Frame', 
                'Intensity', 
                ['Y_Filtered_MA', 'U_Filtered_MA', 'V_Filtered_MA'], 
                f'{output_path}/yuv_ma_filtered.png')
    
# region Setup
def setup():
    args = parse_args()
    date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return args, date_str

def parse_args():
    parser = argparse.ArgumentParser(description="Heart Rate Video Processing")
    parser.add_argument(
        "--video_path",
        type=str,
        # default="data/2025-11-28 17-08-10_smash_it.mp4",
        # default="data/20251128_163018_getting_over_it.mp4",
        # default="data/20251128_164227_unfair_mario.mp4",
        default="data/20251128_165133_geometry_dash.mp4",
        # default="data/base.mp4",
        help="Path to the input video file",
    )
    return parser.parse_args()
# endregion

# region Video Processing
def get_fps(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def extract_avg_rgb(video_path):
    # return
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(int(os.cpu_count())-1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    avgs_rgb = []
    avgs_yuv = []
    face_cascade = CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            face = locate_face(frame, face_cascade)
            if face is not None:
                avgs_rgb.append(avg(frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]))
                avgs_yuv.append(avg(cv2.cvtColor(frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]], cv2.COLOR_BGR2YUV)))
                del face
            del frame
            pbar.update(1)
    cap.release()
    return np.array(avgs_rgb), np.array(avgs_yuv), fps

def locate_face(frame, face_cascade):
    scale = 0.5
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=scale, fy=scale)
    faces = face_cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=3)
    
    if len(faces) == 0:
        return
    x, y, w, h = faces[0]
    return (int(x/scale), int(y/scale), int(w/scale), int(h/scale))

def avg(frame):
    h, w = frame.shape[:2]
    # Define ROI percentages (relative to face bounding box)
    # Forehead (top 20% of face)
    fh_y1 = 0
    fh_y2 = int(0.2 * h)
    fh_x1 = int(0.25 * w)
    fh_x2 = int(0.75 * w)
    
    # Left cheek
    lc_y1 = int(0.3 * h)
    lc_y2 = int(0.6 * h)
    lc_x1 = int(0.1 * w)
    lc_x2 = int(0.4 * w)
    
    # Right cheek
    rc_y1 = int(0.3 * h)
    rc_y2 = int(0.6 * h)
    rc_x1 = int(0.6 * w)
    rc_x2 = int(0.9 * w)
    
    # Extract ROIs
    forehead = frame[fh_y1:fh_y2, fh_x1:fh_x2]
    left_cheek = frame[lc_y1:lc_y2, lc_x1:lc_x2]
    right_cheek = frame[rc_y1:rc_y2, rc_x1:rc_x2]
    
    # Average RGB per ROI
    roi_list = [forehead, left_cheek, right_cheek]
    avg_rgb = np.mean([np.mean(roi.reshape(-1, 3), axis=0) for roi in roi_list], axis=0)
    return avg_rgb
# endregion

# region HRV Analysis
def magnify_colour_ma(ppg, delta=50, n_bg_ma=60, n_smooth_ma=3):
    # Remove slow-moving background component
    ppg = ppg - moving_average_reflect(ppg, n_bg_ma, 0)
    # Smooth the resulting PPG
    ppg = moving_average_reflect(ppg, n_smooth_ma, 0)
    # Remove the NaNs or np.max won't like it
    ppg = np.nan_to_num(ppg)

    scale = np.percentile(np.abs(ppg), 95)
    # Make it have a max delta of delta by normalising by the biggest deviation
    return delta*ppg/scale

def moving_average_reflect(a, n=3, axis=0):
    a = np.asarray(a)
    pad = (n - 1) // 2

    # pad only along chosen axis
    pad_widths = [(0, 0)] * a.ndim
    pad_widths[axis] = (pad, pad)
    a_pad = np.pad(a, pad_widths, mode='reflect')

    # convolution
    kernel = np.ones(n) / n

    # apply convolution along axis
    out = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'),
        axis,
        a_pad
    )

    # remove the pad to get the original length
    slicer = [slice(None)] * a.ndim
    slicer[axis] = slice(pad, pad + a.shape[axis])

    return out[tuple(slicer)]

def sliding_window_fft(x, fs, window_size, step_size):
    # Tried looking at FFT but didn't use it in the end
    N, ch_n = x.shape
    window_samples = math.ceil(window_size * fs)
    step_samples = math.ceil(step_size * fs)
    num_windows = (N - window_samples) // step_samples + 1

    freqs = np.fft.fftfreq(window_samples, 1.0 / fs)
    pos = freqs >= 0
    freqs_pos = freqs[pos]

    hrs = np.zeros((num_windows, ch_n))

    for w in range(num_windows):
        start = w * step_samples
        end = start + window_samples
        window = np.hanning(window_samples)

        for i in range(ch_n):
            segment = x[start:end, i] * window
            X = np.fft.fft(segment)
            magnitudes = np.abs(X[pos]) / window_samples
            max_pos = np.argmax(magnitudes)
            hrs[w, i] = freqs_pos[max_pos] * 60  # Convert to BPM

    return freqs_pos, hrs

def filter_signal(ppg, fs, lowcut=0.7, highcut=4.0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='bandpass')
    # print(ppg.shape)
    for i in range(ppg.shape[1]):
        ppg[:, i] = filtfilt(b, a, ppg[:, i])
    return ppg

def detect_beats(ppg, fs):
    # expected physiological RR intervals
    min_rr = 0.25     # 250 ms (240 BPM)
    max_rr = 1.5      # 1500 ms (40 BPM)

    min_distance = int(min_rr * fs)

    # Peak detection
    peaks, props = find_peaks(
        ppg,
        distance=min_distance,
        prominence=np.std(ppg) * 0.5,   # adjust as needed
        height=np.std(ppg) * 0.2        # threshold relative to noise
    )

    return peaks

def drop_outliers(data):
    # Filter out physiologically implausible RR intervals
    mask = (data > 0.25) & (data < 1.5)
    data = data[mask]
    
    # Remove outliers using Median Absolute Deviation (MAD)
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return data[np.abs(data - med) < 3*mad]

def compute_rr(peaks, fs):
    rr = np.diff(peaks) / fs   # in seconds

    rr_clean = drop_outliers(rr)

    return rr_clean*1000  # return in milliseconds

def compute_hrv_metrics(rr_intervals):
    sdnn = np.std(rr_intervals)
    diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff**2))
    return sdnn, rmssd

def sliding_window_hrv(rr_intervals, window_size=30, step_size=5):
    N = len(rr_intervals)
    window_samples = window_size
    step_samples = step_size
    num_windows = (N - window_samples) // step_samples + 1

    sdnn_values = []
    rmssd_values = []

    for w in range(num_windows):
        start = w * step_samples
        end = start + window_samples
        window_rr = rr_intervals[start:end]
        
        sdnn, rmssd = compute_hrv_metrics(window_rr)
        sdnn_values.append(sdnn)
        rmssd_values.append(rmssd)

    return np.array(sdnn_values), np.array(rmssd_values)
# endregion

# region Save data
def create_plot(x, y, title, xlabel, ylabel, legend, output_path, ylim=None):
    if x is None:
        x = np.arange(y.shape[0])

    colors = ['b', 'g', 'r']

    plt.figure(figsize=(10, 6))
    if y.ndim == 1:
        plt.plot(x, y, color=colors[1])
    else:
        for i in range(y.shape[1]):
            plt.plot(x, y[:, i], color=colors[i])
    plt.title(title)
    plt.xlabel(xlabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def save_csv(file_path, data, header=None):
    """
    Save data to a CSV file.
    
    Args:
        file_path: Path to the output CSV file
        data: 2D array-like data to save
        header: Optional list of column headers
    """
    np.savetxt(file_path, data, delimiter=',', header=','.join(header) if header else '', comments='')
    print(f"Data saved to {file_path}")
# endregion

if __name__ == "__main__":
    main()