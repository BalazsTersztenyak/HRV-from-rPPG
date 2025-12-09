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

    if os.path.exists(f'{output_path}/rgb.csv'):
        print(f"Skipping {path}, already processed.")
        rgb = np.loadtxt(f'{output_path}/rgb.csv', delimiter=',', skiprows=1)
        fps = get_fps(path)
    else:
        rgb, fps = extract_avg_rgb(path)

    save_csv(f'{output_path}/rgb.csv', rgb, header=['Blue', 'Green', 'Red'])

    rgb_ma = magnify_colour_ma(
        np.array(rgb, dtype=np.float64),
        delta=1,
        n_bg_ma=90,
        n_smooth_ma=6
        )

    rgb_ma_filtered = filter_signal(rgb_ma, fps)
    peaks = detect_beats(rgb_ma_filtered[..., 1], fps)
    rr_intervals = compute_rr(peaks, fps)
    
    sdnn, rmssd = compute_hrv_metrics(rr_intervals)
    print(f"SDNN: {sdnn:.2f} ms, RMSSD: {rmssd:.2f} ms")

    sdnn_values, rmssd_values = sliding_window_hrv(rr_intervals)
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

    create_plot(None, 
                rgb,
                'Average RGB', 
                'Frame', 
                'Intensity', 
                ['Blue', 'Green', 'Red'], 
                f'{output_path}/rgb.png')
    
    create_plot(None, 
                rgb_ma_filtered[..., 1],
                'Filtered Magnified Green', 
                'Frame', 
                'Intensity', 
                ['Green_Filtered_MA'], 
                f'{output_path}/g_ma_filtered.png')

    create_plot(None, 
                rgb_ma,
                'Magnified Colour MA', 
                'Frame', 
                'Intensity', 
                ['Blue_MA', 'Green_MA', 'Red_MA'], 
                f'{output_path}/rgb_ma.png')
    
    create_plot(None, 
                rgb_ma_filtered,
                'Filtered Magnified Colour MA', 
                'Frame', 
                'Intensity', 
                ['Blue_Filtered_MA', 'Green_Filtered_MA', 'Red_Filtered_MA'], 
                f'{output_path}/rgb_ma_filtered.png')
    
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
    avgs = []
    face_cascade = CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            face = locate_face(frame, face_cascade)
            if face is not None:
                avgs.append(avg(frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]))
                del face
            del frame
            pbar.update(1)
    cap.release()
    return np.array(avgs), fps

def locate_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return
    return faces[0]

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
    min_rr = 0.25     # 250 ms
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

################# Delete below this line ####################

# # region Main
# def main_old():
#     return
#     args, date_str = setup()
    
#     face_frames, face_frames_yuv, fps = extract_avg_rgb(args.video_path)
#     # save_frames_to_video(face_frames, output_path=f'face_cutout_{date_str}.mp4', fps=fps)
#     ppg_rgb = avg_per_channels(face_frames)
#     ppg_yuv = avg_per_channels(face_frames_yuv)
#     del face_frames  # Free up memory
#     del face_frames_yuv
#     save_csv(f'ppg_rgb_output_{date_str}.csv', ppg_rgb, header=['Blue', 'Green', 'Red'])
#     save_csv(f'ppg_yuv_output_{date_str}.csv', ppg_yuv, header=['Y', 'U', 'V'])
#     ppg_rgb_ma = magnify_colour_ma(
#         np.array(ppg_rgb, dtype=np.float64),
#         delta=1,
#         n_bg_ma=90,
#         n_smooth_ma=6
#         )
#     ppg_yuv_ma = magnify_colour_ma(
#         np.array(ppg_yuv, dtype=np.float64),
#         delta=1,
#         n_bg_ma=90,
#         n_smooth_ma=6
#         )
#     save_csv(f'ppg_rgb_ma_output_{date_str}.csv', ppg_rgb_ma, header=['Blue_MA', 'Green_MA', 'Red_MA'])
#     save_csv(f'ppg_yuv_ma_output_{date_str}.csv', ppg_yuv_ma, header=['Y_MA', 'U_MA', 'V_MA'])
#     fft_freqs, rgb_fft_ma, max_freqs_ma = compute_fft(ppg_rgb_ma, fps)
#     print('Max frequencies (MA):', max_freqs_ma)
#     filtered_ppg_rgb_ma = filter_signal(ppg_rgb_ma, fps)
#     fft_freqs, filtered_fft_rgb_ma, max_freqs_filtered_ma = compute_fft(filtered_ppg_rgb_ma, fps)
#     print('Max frequencies (Filtered MA):', max_freqs_filtered_ma)
#     save_csv(f'ppg_fft_freqs_{date_str}.csv', fft_freqs, header=['Frequency_Hz'])
#     save_csv(f'ppg_rgb_ma_fft_output_{date_str}.csv', rgb_fft_ma, header=['Blue_FFT', 'Green_FFT', 'Red_FFT'])
#     fft_freqs, rgb_fft, max_freqs = compute_fft(ppg_rgb, fps)
#     print('Max frequencies:', max_freqs)
#     save_csv(f'ppg_rgb_fft_output_{date_str}.csv', rgb_fft, header=['Blue_FFT', 'Green_FFT', 'Red_FFT'])
    
#     print(rgb_fft_ma.shape)
#     plt.figure(figsize=(10, 6))
#     plt.plot(fft_freqs, rgb_fft_ma)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
#     plt.title('FFT of PPG Signal')
#     plt.legend(['Blue', 'Green', 'Red'])
#     plt.savefig(f'ppg_rgb_fft_plot_{date_str}.png')

#     print(filtered_fft_rgb_ma.shape)
#     plt.figure(figsize=(10, 6))
#     plt.plot(fft_freqs, filtered_fft_rgb_ma)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
#     plt.title('FFT of Filtered PPG Signal')
#     plt.legend(['Blue', 'Green', 'Red'])
#     plt.savefig(f'ppg_rgb_fft_plot_{date_str}.png')
#     # plt.show()

# def main2():
#     args, date_str = setup()
#     video_array, fps = load_video_to_array(args.video_path)
#     print(f"Video loaded with shape: {video_array.shape}, FPS: {fps}")
#     # face_frames, face_frames_yuv = process_video_array(video_array)
#     face_frames = parallel_face_detection(args.video_path)
#     # face_frames = parallel_face_detection_from_path(args.video_path)
#     rgb = [face[0] for face in face_frames]
#     yuv = [face[1] for face in face_frames]
#     del face_frames
#     ppg_rgb = avg_per_channels(rgb)
#     ppg_yuv = avg_per_channels(yuv)
    
#     ppg_rgb_ma = magnify_colour_ma(
#         np.array(ppg_rgb, dtype=np.float64),
#         delta=1,
#         n_bg_ma=90,
#         n_smooth_ma=6
#         )
#     ppg_yuv_ma = magnify_colour_ma(
#         np.array(ppg_yuv, dtype=np.float64),
#         delta=1,
#         n_bg_ma=90,
#         n_smooth_ma=6
#         )
#     del ppg_rgb, ppg_yuv

#     create_plot(np.arange(ppg_rgb_ma.shape[0])/fps, ppg_rgb_ma, 'PPG Signal (RGB MA)', 'Time (s)', 'Amplitude', ['Blue_MA', 'Green_MA', 'Red_MA'], f'ppg_rgb_ma_plot_{date_str}.png', ylim=(-0.02, 0.02))
#     create_plot(np.arange(ppg_yuv_ma.shape[0])/fps, ppg_yuv_ma, 'PPG Signal (YUV MA)', 'Time (s)', 'Amplitude', ['Y_MA', 'U_MA', 'V_MA'], f'ppg_yuv_ma_plot_{date_str}.png', ylim=(-0.02, 0.02))

#     freqs, rgb_fft_ma, max_freqs_ma = compute_fft(ppg_rgb_ma, fps)
#     print('Max frequencies RGB (MA):', max_freqs_ma)
#     freqs, yuv_fft_ma, max_freqs_ma = compute_fft(ppg_yuv_ma, fps)
#     print('Max frequencies YUV (MA):', max_freqs_ma)

#     filtered_rgb_ma = filter_signal(ppg_rgb_ma, fps)
#     freqs, filtered_rgb_fft_ma, max_freqs_filtered_rgb_ma = compute_fft(filtered_rgb_ma, fps)
#     print('Max frequencies RGB (Filtered MA):', max_freqs_filtered_rgb_ma)
#     filtered_yuv_ma = filter_signal(ppg_yuv_ma, fps)
#     freqs, filtered_yuv_fft_ma, max_freqs_filtered_yuv_ma = compute_fft(filtered_yuv_ma, fps)
#     print('Max frequencies YUV (Filtered MA):', max_freqs_filtered_yuv_ma)

#     create_plot(freqs, rgb_fft_ma, 'FFT of PPG Signal (RGB MA)', 'Frequency (Hz)', 'Magnitude', ['Blue', 'Green', 'Red'], f'ppg_rgb_fft_plot_{date_str}.png')
#     create_plot(freqs, filtered_rgb_fft_ma, 'FFT of Filtered PPG Signal (RGB MA)', 'Frequency (Hz)', 'Magnitude', ['Blue', 'Green', 'Red'], f'ppg_rgb_filtered_fft_plot_{date_str}.png')
#     create_plot(freqs, yuv_fft_ma, 'FFT of PPG Signal (YUV MA)', 'Frequency (Hz)', 'Magnitude', ['Y', 'U', 'V'], f'ppg_yuv_fft_plot_{date_str}.png')
#     create_plot(freqs, filtered_yuv_fft_ma, 'FFT of Filtered PPG Signal (YUV MA)', 'Frequency (Hz)', 'Magnitude', ['Y', 'U', 'V'], f'ppg_yuv_filtered_fft_plot_{date_str}.png')
    
#     all_maxes = np.concatenate((
#         max_freqs_filtered_rgb_ma,
#         max_freqs_filtered_yuv_ma
#     ))
#     print(np.mean(all_maxes))

#     freqs, magnitudes_rgb = sliding_window_fft(filtered_rgb_ma, fps, window_size=10, step_size=0.5)
#     freqs, magnitudes_yuv = sliding_window_fft(filtered_yuv_ma, fps, window_size=10, step_size=0.5)

#     create_plot(None, magnitudes_rgb, 'Sliding Window HR (RGB)', 'Time', 'HR', ['Blue', 'Green', 'Red'], f'sliding_window_rgb_fft_{date_str}.png')
#     create_plot(None, magnitudes_yuv, 'Sliding Window HR (YUV)', 'Time', 'HR', ['Y', 'U', 'V'], f'sliding_window_yuv_fft_{date_str}.png')
# # endregion


# # region Load Video
# def get_video_properties(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Video file not found: {path}")
#     cap = cv2.VideoCapture(path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()
#     return frame_count, fps, frame_width, frame_height

# def load_video_to_array(video_path):
#     total_frames, fps, frame_width, frame_height = get_video_properties(video_path)
#     video_array = np.zeros((total_frames, frame_height, frame_width, 3), dtype=np.uint8)
#     cap = cv2.VideoCapture(video_path)
#     with tqdm(total=total_frames, desc="Loading video") as pbar:
#         frame_idx = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             video_array[frame_idx] = frame
#             frame_idx += 1
#             pbar.update(1)
#     cap.release()
#     return video_array, fps
# # endregion

# # region Processing

# def process_video_array(video_array):
#     return
#     total_frames = video_array.shape[0]
#     face_frames = []
#     face_frames_yuv = []
#     avgs = []
#     for i in range(total_frames):
#         frame = video_array[i]
#         face = locate_face(frame)
#         if face is not None:
#             face_frames.append(face)
#             face_frames_yuv.append(cv2.cvtColor(face, cv2.COLOR_BGR2YUV))
#     return face_frames, face_frames_yuv

# def parallel_face_detection(video_array):
#     max_workers = int(os.cpu_count()*0.8)

#     results = {}

#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = {executor.submit(locate_face, video_array[i]): i for i in range(video_array.shape[0])}

#         with ProgressBar(total=video_array.shape[0], desc="Parallel face detection") as pbar:
#             for future in as_completed(futures):
#                 idx = futures[future]
#                 try:
#                     results[idx] = future.result()
#                 except Exception as e:
#                     print(f"Error processing frame {idx}: {e}")
#                 pbar.update(1)

#     frames = [results[key] for key in sorted(results.keys())]

#     return frames

# def parallel_face_detection_from_path(video_path):
#     cap = cv2.VideoCapture(video_path)
#     cv2.setUseOptimized(True)
#     cv2.setNumThreads(os.cpu_count()-1)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     # frames = []
#     cascade = CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")

#     with tqdm(total=total_frames, desc="Reading video frames") as pbar:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             rgb, yuv = locate_face(frame, cascade)
#             # frames.append(frame)
#             pbar.update(1)
#     cap.release()

#     # video_array = np.array(frames)
#     # del frames


# def avg_per_channels(frames):
#     rgb = np.zeros((len(frames), 3))
#     for i, frame in enumerate(frames):
#         rgb[i, 0] = np.mean(frame[:, :, 0])  # Blue channel
#         rgb[i, 1] = np.mean(frame[:, :, 1])  # Green channel
#         rgb[i, 2] = np.mean(frame[:, :, 2])  # Red channel
#     return rgb


# # endregion

# def save_frames_to_video(frames, output_path='output.mp4', fps=30):
#     """
#     Save a list of NumPy arrays (images) as a video file.
    
#     Args:
#         frames: List of NumPy arrays (images)
#         output_path: Output video file path
#         fps: Frames per second for the output video
#     """
#     if not frames:
#         print("No frames to save!")
#         return
    
#     # Get dimensions from first frame
#     height, width = frames[0].shape[:2]
    
#     # Define the codec and create VideoWriter object
#     fourcc = VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264
#     out = VideoWriter(output_path, fourcc, fps, (width, height))
    
#     for frame in frames:
#         # Ensure frame is in BGR format (OpenCV uses BGR, not RGB)
#         if len(frame.shape) == 2:  # Grayscale
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#         # elif frame.shape[2] == 3:
#         #     # If your arrays are in RGB format, convert to BGR
#         #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
#         # Ensure frame has correct dimensions
#         if frame.shape[:2] != (height, width):
#             frame = cv2.resize(frame, (width, height))
        
#         out.write(frame)
    
#     out.release()
#     print(f"Video saved to {output_path}")






# def compute_fft(x, fs):
#     N, ch_n = x.shape
#     X = np.zeros((N, ch_n), dtype=complex)
#     window = np.hanning(N)
#     for i in range(ch_n):
#         X[:, i] = np.fft.fft(x[:, i] * window)
#     freqs = np.fft.fftfreq(N, 1.0 / fs)

#     pos = freqs >= 0
#     freqs_pos = freqs[pos]
#     magnitudes = np.abs(X[pos, :]) / N

#     max_pos = np.argmax(magnitudes, axis=0)
#     max = freqs_pos[max_pos]
#     return freqs_pos, magnitudes, max*60  # Convert to BPM

if __name__ == "__main__":
    main()