######################
### USER VARIABLES ###
######################

# Square this because I'll be lazily not-square-rooting later
chroma_similarity = 10**2

video_width = 1920
video_height = 1080

capture_device = 0

buffer_size = 1024

#######################
### /USER VARIABLES ###
#######################

# A few other variables
display_mode = 'normal'
fps_display = True

processing_mode = 'average_ppg'

# Libraries
import cv2 as cv
import numpy as np
import time
import csv
import random
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy import signal
from scipy.ndimage import uniform_filter1d

# Set up command line argument to process a file rather than live video
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="Process a video file rather than a capture device.", metavar="FILE")
parser.add_argument('-w','--welch', dest='welch_flag', action=argparse.BooleanOptionalAction,
                    help='Compute heart rate using the Welch estimator.')
args = parser.parse_args()

# Write some data to files
def csv_xyz(filename, data, names):
    csv_begin(filename, names)
    csv_append(filename, data)

def mouseRGB(event, x, y, flags, params):
    global skin_chroma
    if event == cv.EVENT_LBUTTONDOWN:
        skin_chroma = np.array(cv.cvtColor(np.array([[frame[y,x]]]), cv.COLOR_BGR2YUV), dtype=np.float32)[0,0,1:3]
        print('RGB = ', frame[y,x], 'chroma = ', skin_chroma)

def chroma_key(frame, chroma):
    key = frame[:,:,1:3] - chroma
    key = np.less(np.sum(np.square(key), axis=2), chroma_similarity)
    return key

def chroma_key_display(frame, chroma):
    """
    Convenience function to display the chroma key
    """
    key = chroma_key(frame, chroma)
    return (key*255).astype(np.uint8)

def moving_average(a, n=3, axis=None):
    # If it's not None, we're not gonna flatten the array...
    if axis is not None:
        # ...so temporarily swap the axis to the end
        ret = np.swapaxes(a, 0, axis)
    else:
        ret = a
    # take the cumulative sum of the input vector
    ret = np.cumsum(ret, axis=axis)
    # subtract the cumsum, offset by n, to get the moving average via kludge
    ret[n:,...] = ret[n:,...] - ret[:-n,...]
    # Concatenate together 0 ..the numbers... 0 0 to pad it to the original length
    ret = np.concatenate((
        # Following what R does, return fewer 0s at the start if n is even...
        np.zeros((int(np.floor((n-1)/2)), *ret.shape[1:])),
        # ...then some numbers...
        ret[(n - 1):,...] / n,
        # ...then more 0s at the end if n is even (both equal if odd!)
        np.zeros((int(np.ceil((n-1)/2)), *ret.shape[1:]))
    ))
    # Swap the axis back if we swapped it at the start
    if axis is not None:
        ret = np.swapaxes(ret, 0, axis)
    return ret

def average_keyed(frame, key):
    """
    Return the average YUV of the pixels which are True in key.
    Args:
        frame: a numpy array containing the frame
        key: a numpy array of booleans
    Returns:
        A numpy array of [Y, U, V]
    """
    output = np.mean(frame[key], axis=0)
    return output

def csv_begin(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def csv_append(filename, data):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def magnify_colour_ma(ppg, delta=50, n_bg_ma=60, n_smooth_ma=3):
    # Remove slow-moving background component
    ppg = ppg - moving_average(ppg, n_bg_ma, 0)
    # Smooth the resulting PPG
    ppg = moving_average(ppg, n_smooth_ma, 0)
    # Remove the NaNs or np.max won't like it
    ppg = np.nan_to_num(ppg)
    # Make it have a max delta of delta by normalising by the biggest deviation
    return delta*ppg/np.max(np.abs(ppg))

def magnify_colour_ma_masked(ppg, mask, delta=50, n_bg_ma=60, n_smooth_ma=3):
    # Remove slow-moving background component
    ppg = ppg - moving_average(ppg, n_bg_ma, 0)
    mask = moving_average(mask, n_bg_ma, 0)
    # Smooth the resulting PPG
    ppg = moving_average(ppg, n_smooth_ma, 0)
    mask = moving_average(mask, n_smooth_ma, 0)
    # Expand the mask to allow it to be used to, er, mask the ppg
    # Remove any pixels in ppg that go to zero at any point in the windows, found because the mask which has been
    # equivalently moving-averaged above drops below 1
    ppg = np.where(mask[:,:,:, np.newaxis] == 1., ppg, np.zeros_like(ppg))
    # Remove the NaNs or np.max won't like it
    ppg = np.nan_to_num(ppg)
    # Set the Y component to 0 (could've just done calcs on only U and V earlier, but I'm lazy)
    ppg[:,:,:,0] = 0
    # Make it have a max delta of delta by normalising by the biggest deviation
    return delta*ppg/np.max(np.abs(ppg))

def Welch_cpu(filename, bvps, fps, nfft=8192):
    """
    This function computes Welch's method for spectral density estimation on CPU.
    Args:
        bvps(float32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float32): frames per seconds.
        nfft (int32): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    bvps = np.transpose(np.array(bvps, dtype=np.float32))
    for i in range(128, bvps.shape[0]-128):
        print(i, bvps.shape)
        t = (i+128)/fps
        # -- periodogram by Welch
        F, P = signal.welch(bvps[(i-128):(i+128)], nperseg=256,
                                noverlap=200, fs=fps, nfft=nfft)
        # write t, F, P to our CSV
        towrite = np.concatenate((np.full((F.shape), t)[:,None],F[:,None],P[:,None]), axis=1)
        csv_append(filename, towrite)

def display_image(img, window='StattoBPM'):
    if len(img.shape) == 3:
        cv.imshow(window, cv.cvtColor(img.astype(np.uint8), cv.COLOR_YUV2BGR))
    elif len(img.shape) == 2:
        cv.imshow(window, img.astype(np.uint8)*255)

def keypress_action(keypress):
    global display_mode, fps_display
    # Default keypress is -1, which means 'do nothing' so skip the below
    if keypress != -1:
        # If the keypress is in the dictionary of display modes...
        if keypress == ord('a'):
            # if it's not already in that mode, put it in that mode
            if display_mode != 'alpha_channel':
                display_mode = 'alpha_channel'
            # if it is already in that mode, put things back to normal
            else:
                display_mode = 'normal'
            print('Display mode ' + display_mode + ' activated')
        # If zero is pressed, record an in breath
        elif keypress == 48:
            csv_append('data/breaths.csv', [[t, 0]])
        # If . is pressed, record an out breath
        elif keypress == 46:
            csv_append('data/breaths.csv', [[t, 1]])
        # If you press f, enables/disables fps display
        elif keypress == ord('f'):
            fps_display = not(fps_display)
            print('FPS display set to', fps_display)
        # Press Esc or q to exit
        elif keypress==27 or keypress == ord('q'):
            print('Goodbye!')
            exit()
        else:
            print('You pressed %d (0x%x), LSB: %d (%s)' % (keypress, keypress, keypress % 256,
                    repr(chr(keypress%256)) if keypress%256 < 128 else '?'))

# If no filename was specified, open a capture device
if(args.filename is None):
    # define a video capture object
    vid = cv.VideoCapture(capture_device)
    if not vid.isOpened():
        print('Error: Cannot open camera')
        exit()
    else:
        print('Initialising camera...')
        print('  Default frame size: ' + str(vid.get(cv.CAP_PROP_FRAME_WIDTH)) + '×' + str(vid.get(cv.CAP_PROP_FRAME_HEIGHT)))
    # Override default frame size for capture device, which is often 640x480
    vid.set(cv.CAP_PROP_FRAME_WIDTH, video_width)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, video_height)
    # So we're using a capture device, not a file
    is_video_file = False
else:
    vid = cv.VideoCapture(args.filename)
    if (vid.isOpened() == False):
        print("Error opening video file " + args.filename)
        exit()
    # get fps from the video
    fps = vid.get(cv.CAP_PROP_FPS)
    # And total frame count
    total_frames = vid.get(cv.CAP_PROP_FRAME_COUNT)
    # So we're using a file, not a capture device
    is_video_file = True

ppg_yuv = []
ppg_rgb = []
times  = []
# Instantiate skin_chroma
skin_chroma = np.zeros(2, dtype=np.float32)

# Instantiate the window we're going to display BPM in
cv.namedWindow('StattoBPM')
# Add the ability to detect mouse clicks
cv.setMouseCallback('StattoBPM', mouseRGB)

# If this is a video file, start a loop to set the chroma key in advance of processing
print('Displaying random frames. Click to set chroma. Press A to toggle chroma key view and O once you\'re ready to go!')
if is_video_file:
    while True:
        # Get a random frame somewhere near the middle of the video
        random_frame = random.randrange(int(total_frames/4), int(3*total_frames/4))
        vid.set(1, random_frame)
        ret, frame = vid.read()
        frame_cp = np.array(cv.cvtColor(frame, cv.COLOR_BGR2YUV), dtype=np.float32)
        if display_mode == 'alpha_channel':
            cv.imshow(
                'StattoBPM',
                chroma_key_display(frame_cp, skin_chroma)
            )
        else:
            cv.imshow(
                'StattoBPM',
                frame
            )
        keypress = cv.waitKey(100)
        if keypress == ord('o'):
            print('OK, chroma value set! Let\'s compute!')
            # Set to the first frame so the whole video gets processed
            vid.set(1, 0)
            break
        keypress_action(keypress)

# First loop: analysis
print('First pass: analysing video...')
i = 0
t0 = time.time()
while True:
    # Get a frame from the video capture device or file
    ret, frame = vid.read()
    # Store the current time in the buffer
    times.append(time.time() - t0)
    # And calculate the fps, either of processing or capture depending on device
    # (Don't calculate on the first pass through the loop to avoid dividing by zero)
    if i > 0:
        fps_calc = len(times) / (times[-1] - times[0])
    # If we're using a capture device, the fps is given by the above, rather than specified beforehand
    if not is_video_file:
        fps = fps_calc
        t = times[-1]
    # If we're using a video file, then the time we are through is frame / fps
    else:
        t = i/fps
    if i > 0 and fps_display and i % 100 == 0:
        print('Frame', i, 'of', int(total_frames), '  |  FPS:', np.round(fps_calc, 3))
    # If ret is false, it usually means 'video file is over', but it's an error either way, so exit the loop
    if not ret:
        print('Pass 1 complete!')
        break

    frame_cp = np.array(frame, dtype=np.float32)
    frame_yuv = np.array(cv.cvtColor(frame, cv.COLOR_BGR2YUV), dtype=np.float32)
    skin_key = chroma_key(frame_yuv, skin_chroma)
    ppg_rgb.append(average_keyed(frame_cp, skin_key))
    ppg_yuv.append(average_keyed(frame_yuv, skin_key))

    if display_mode == 'alpha_channel':
        cv.imshow(
            'StattoBPM',
            chroma_key_display(frame_yuv, skin_chroma)
        )
    else:
        cv.imshow('StattoBPM', frame)
    keypress_action(cv.waitKey(1))
    i = i + 1

# Calculations
print('First pass completed. Doing calculations...')

ppg_rgb_ma = magnify_colour_ma(
    np.array(ppg_rgb, dtype=np.float64),
    delta=1,
    n_bg_ma=90,
    n_smooth_ma=6
    )
ppg_yuv_ma = magnify_colour_ma(
    np.array(ppg_yuv, dtype=np.float64),
    delta=1,
    n_bg_ma=90,
    n_smooth_ma=6
    )
# 'white', averaging RGB
ppg_w_ma = np.mean(ppg_rgb_ma, axis=1)

outdir = 'output-data-' + args.filename
counter = 1
mypath = outdir + '-' + str(counter)
while os.path.exists(mypath):   
    counter += 1
    mypath = outdir + '-' + str(counter)

os.makedirs(mypath)

csv_xyz(os.path.join(mypath, 'ppg-rgb.csv'), np.array(ppg_rgb, dtype=np.float64), ['b', 'g', 'r'])
csv_xyz(os.path.join(mypath, 'ppg-rgb-ma.csv'), np.array(ppg_rgb_ma, dtype=np.float64), ['b', 'g', 'r'])
csv_xyz(os.path.join(mypath, 'ppg-yuv.csv'), np.array(ppg_yuv, dtype=np.float64), ['y', 'u', 'v'])
csv_xyz(os.path.join(mypath, 'ppg-yuv-ma.csv'), np.array(ppg_yuv_ma, dtype=np.float64), ['y', 'u', 'v'])

with open(os.path.join(mypath, 'chroma-key.txt'), 'w') as f:
    f.write(str(skin_chroma))

matplotlib.use('TKAgg')

def normalise(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

data0 = {'time': np.array(range(int(total_frames)))/fps,
        'red': np.array(ppg_rgb, dtype=np.float64)[:,2],
        'green': np.array(ppg_rgb, dtype=np.float64)[:,1],
        'blue': np.array(ppg_rgb, dtype=np.float64)[:,0]}

fig, ax = plt.subplots()
ax.plot('time', 'red', data=data0, color='red')
ax.plot('time', 'green', data=data0, color='green')
ax.plot('time', 'blue', data=data0, color='blue')
ax.set_xlabel('time')
ax.set_ylabel('RGB')
plt.show()

data = {'time': np.array(range(100,int(total_frames)-100))/fps,
        'red': ppg_rgb_ma[100:-100,2],
        'green': ppg_rgb_ma[100:-100,0],
        'blue': ppg_rgb_ma[100:-100,1]}

fig, ax = plt.subplots()
ax.plot('time', 'red', data=data, color='red')
ax.plot('time', 'green', data=data, color='green')
ax.plot('time', 'blue', data=data, color='blue')
ax.set_xlabel('time')
ax.set_ylabel('RGB')
plt.show()

data2 = {'time': np.array(range(100,int(total_frames)-100))/fps,
        'luminance': ppg_yuv_ma[100:-100,0],
        'colour-u': ppg_yuv_ma[100:-100,1],
        'colour-v': ppg_yuv_ma[100:-100,2]}

fig, ax = plt.subplots()
ax.plot('time', 'luminance', data=data2, color='black')
ax.plot('time', 'colour-u', data=data2, color='green')
ax.plot('time', 'colour-v', data=data2, color='magenta')
ax.set_xlabel('time')
ax.set_ylabel('YUV')
plt.show()

# Reopen the video
vid = cv.VideoCapture(args.filename)
if vid.isOpened() == False:
    print("Error opening video file " + args.filename)
    exit()
# So we're using a file, not a capture device
is_video_file = True

# Second loop: adding stuff
print('Second pass: saving results!')
frames_path = os.path.join(mypath, 'frames-uvw')
os.makedirs(frames_path)
times = []
i = 0
t0 = time.time()
while True:
    # Get a frame from the video capture device or file
    ret, frame = vid.read()
    # Store the current time in the buffer
    times.append(time.time() - t0)
    # And calculate the fps, either of processing or capture depending on device
    # (Don't calculate on the first pass through the loop to avoid dividing by zero)
    if i > 0:
        fps_calc = len(times) / (times[-1] - times[0])
    # If we're using a capture device, the fps is given by the above, rather than specified beforehand
    if not is_video_file:
        fps = fps_calc
        t = times[-1]
    # If we're using a video file, then the time we are through is frame / fps
    else:
        t = i/fps
    if i > 0 and fps_display and i % 100 == 0:
        print('Frame', i, 'of', int(total_frames), '  |  FPS:', np.round(fps_calc, 3))
    # If ret is false, it usually means 'video file is over', but it's an error either way, so exit the loop
    if not ret:
        print('Pass 2 complete!')
        break
    frame_yuv = np.array(cv.cvtColor(frame, cv.COLOR_BGR2YUV), dtype=np.float32)
    skin_key = chroma_key(frame_yuv, skin_chroma)
    colours_w = np.moveaxis(np.array([np.zeros_like(skin_key), skin_key * ppg_w_ma[i], skin_key * ppg_w_ma[i]]), 0, -1)
    output_uv_w = cv.cvtColor((frame_yuv + colours_w[0:1080, 0:1920]*10000).astype(np.uint8), cv.COLOR_YUV2BGR)
    cv.imshow('StattoBPM', output_uv_w)
    cv.imwrite(os.path.join(frames_path, 'uvw_magnified-'+f'{i:05}'+'.png'), output_uv_w)
    keypress_action(cv.waitKey(1))
    i = i + 1

# Welch estimate of heart rate
if args.welch_flag:
    print('Computing Welch estimate...')
    Welch_cpu(os.path.join(mypath, 'welch.csv'), ppg_w_ma, fps)
    print('Welch estimate complete!')

print('All done!')