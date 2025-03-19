import numpy as np
from pylsl import StreamInlet, resolve_streams
from scipy.signal import welch, butter, filtfilt
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# ----- Configurations ----- #
SAMPLING_RATE = 250  # Hz
WINDOW_SIZE_SEC = 2  # seconds
PLOT_HISTORY_SEC = 10  # seconds
CHANNEL_INDEX = 0  # first channel

ALPHA_BAND = (8, 12)
UPDATE_INTERVAL_MS = 100  # plot and processing update interval

PLOT_HISTORY_SEC = 10  # Display last 10 seconds
WINDOW_SIZE_SEC = 2     # 2-second window

THRESHOLD_ALPHA = 0.4

# ----- Initialize LSL ----- #
print("Looking for an EEG stream...")
streams = resolve_streams()
print("Found {} streams.".format(len(streams)))
for i, stream in enumerate(streams):
    print("Stream {} name: {}".format(i, stream.name()))

# Get user input of what steam is EEG
stream_num = input("Enter the number of the stream you want to plot: ")
inlet = StreamInlet(streams[int(stream_num)])


# ----- Data Buffers ----- #
buffer_size = int(PLOT_HISTORY_SEC * SAMPLING_RATE)
eeg_buffer = deque(maxlen=buffer_size)
time_buffer = deque(maxlen=buffer_size)

alpha_power_history = deque(maxlen=buffer_size)
power_time_buffer = deque(maxlen=buffer_size)

# ----- PyQtGraph Setup ----- #
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# Initialize Qt Application
app = QtWidgets.QApplication([])

win = pg.GraphicsLayoutWidget(title="Live EEG Alpha Power Demo")
win.resize(1000, 600)

# ---- Plot Configurations ----
# EEG plot
p1 = win.addPlot(title="EEG Signal (Single Channel)")
p1.setLabel('left', 'Amplitude', units='uV')
p1.setLabel('bottom', 'Time', units='s')
p1.setYRange(-500, 500)
p1.setXRange(-PLOT_HISTORY_SEC, 0)

eeg_curve = p1.plot()

window_marker = pg.LinearRegionItem(
    [-WINDOW_SIZE_SEC, 0],
    movable=False,
    brush=(50, 50, 200, 50)
)
p1.addItem(window_marker)

# Alpha power plot
win.nextRow()
p2 = win.addPlot(title="Relative Alpha Power (8-12 Hz)")
p2.setLabel('left', 'Relative Power')
p2.setLabel('bottom', 'Time', units='s')
p2.setYRange(0, 1)
p2.setXRange(-PLOT_HISTORY_SEC, 0)

# Add threshold line to alpha power plot
threshold_line = pg.InfiniteLine(pos=THRESHOLD_ALPHA, angle=0, pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
p2.addItem(threshold_line)

# Status label: Eyes Open / Closed
status_text = pg.TextItem(text='State: Unknown', anchor=(0, 1), color='w')
p2.addItem(status_text)
status_text.setPos(-PLOT_HISTORY_SEC + 0.5, 0.95)  # Position in the top-left corner


alpha_curve = p2.plot()

win.show()

# ----- Helper Functions ----- #
def compute_relative_alpha_power(signal, fs, alpha_band):
    f, psd = welch(signal, fs=fs, nperseg=fs//2)
    total_power = np.trapezoid(psd, f)
    idx_band = np.logical_and(f >= alpha_band[0], f <= alpha_band[1])
    alpha_power = np.trapezoid(psd[idx_band], f[idx_band])
    relative_power = alpha_power / total_power if total_power > 0 else 0
    return relative_power

def create_bandpass_filter(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, b, a):
    return filtfilt(b, a, data)


# ----- Update Function ----- #
def update():
    global eeg_buffer, time_buffer, alpha_power_history, power_time_buffer

    # Pull chunk from LSL
    chunk, timestamps = inlet.pull_chunk(timeout=0.1)
    if timestamps:
        for sample, ts in zip(chunk, timestamps):
            eeg_buffer.append(sample[CHANNEL_INDEX])
            time_buffer.append(ts)

    # Convert buffers to arrays
    if len(time_buffer) < WINDOW_SIZE_SEC * SAMPLING_RATE:
        return  # not enough data yet

    times = np.array(time_buffer)
    eeg_data_raw = np.array(eeg_buffer)


    # Apply bandpass filter to the entire EEG buffer
    # Filter coefficients (Butterworth 1-20 Hz)
    b_bandpass, a_bandpass = create_bandpass_filter(1, 20, SAMPLING_RATE, order=4)
    eeg_data_filtered = apply_bandpass_filter(eeg_data_raw, b_bandpass, a_bandpass)

    # Process the last WINDOW_SIZE_SEC of filtered data for alpha power
    current_window_data = eeg_data_filtered[-int(WINDOW_SIZE_SEC * SAMPLING_RATE):]
    relative_alpha = compute_relative_alpha_power(current_window_data, SAMPLING_RATE, ALPHA_BAND)

    alpha_power_history.append(relative_alpha)
    power_time_buffer.append(times[-1])

    # Time axis shifted so last point is at 0
    eeg_curve.setData(times - times[-1], eeg_data_filtered)
    alpha_curve.setData(np.array(power_time_buffer) - times[-1], alpha_power_history)

    # Move the region marker
    window_marker.setRegion([-WINDOW_SIZE_SEC, 0])

    # Update Alpha Power plot
    alpha_curve.setData(np.array(power_time_buffer) - times[-1], alpha_power_history)

    # Update Eyes Open / Closed status
    if relative_alpha > THRESHOLD_ALPHA:
        status = "Eyes CLOSED"
        status_color = 'g'  # Green if closed
    else:
        status = "Eyes OPEN"
        status_color = 'r'  # Red if open

    status_text.setText(f'State: {status}', color=status_color)


# ----- Timer for real-time updates ----- #
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_INTERVAL_MS)

# Start the Qt event loop
QtWidgets.QApplication.instance().exec()
