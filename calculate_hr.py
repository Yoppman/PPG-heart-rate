# calculate_hr.py

import pyPPG
from pyPPG import PPG
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.ppg_sqi as SQI
import numpy as np
from dotmap import DotMap
from scipy.signal import butter, filtfilt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def load_data_from_array(data, fs):
    """
    Mimic the load_data function to create a signal object from a numpy array.
    """
    s = DotMap()
    s.start_sig = 0
    s.end_sig = len(data)
    s.v = data
    s.fs = fs
    s.name = 'Real-time PPG Signal'
    return s



def high_pass_filter(data, cutoff, fs, order=4):
    """
    Apply a high-pass Butterworth filter to the data.
    
    :param data: The input signal.
    :param cutoff: The cutoff frequency of the high-pass filter.
    :param fs: The sampling frequency of the signal.
    :param order: The order of the filter.
    :return: The filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_heart_rate(data, fs, filtering=True, fL=0.5, fH=12.0, order=4, sqi_threshold=0.3):
    """
    Calculate the average heart rate from a PPG signal with SQI filtering.

    :param data: The PPG signal data as a numpy array.
    :param fs: Sampling frequency of the PPG signal.
    :param filtering: Whether to filter the PPG signal.
    :param fL: Lower cutoff frequency for filtering.
    :param fH: Upper cutoff frequency for filtering.
    :param order: Filter order.
    :param sqi_threshold: Minimum SQI value to consider a segment valid.
    :return: Average heart rate (BPM) and preprocessed PPG signal.
    """
    # Ensure data is a numpy array
    data = np.array(data)

    # Step 1: Load the PPG signal from array
    s = load_data_from_array(data, fs)

    # Step 2: Preprocess the PPG signal (filtering and derivatives)
    prep = PP.Preprocess(fL=fL, fH=fH, order=order,
                         sm_wins={'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10})
    s.ppg, s.vpg, s.apg, s.jpg = prep.get_signals(s=s)

    # Step 3: Create a PPG class
    s = PPG(s=s, check_ppg_len = False)


    # Step 4: Extract fiducial points
    fpex = FP.FpCollection(s=s)

    fiducials = fpex.get_fiducials(s=s)
    print(f"sp len : {len(fiducials.sp)}")

    total_seconds = len(data)/ 100
    print(f"Total second {total_seconds}")

    print( f"Raw IPM: {len(fiducials.sp)} * {int(60/total_seconds)}")


    # Step 5: Calculate SQI for the PPG signal
    sqi = SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fiducials.sp)

    # Step 6: Filter out low-quality segments based on SQI
    valid_indices = np.where(sqi > sqi_threshold)[0]  # Indices with SQI > threshold
    valid_sp = fiducials.sp.values[valid_indices]  # Filter systolic peaks with valid SQI


    # Calculate IPM
    total_seconds = len(data) / fs
    filtered_ipm = len(valid_sp) * (60 / total_seconds)

    # Adjust IPM based on average interval
    if len(valid_sp) > 1:
        avg_interval = np.mean(np.diff(fiducials.sp) / fs)  # Average interval in seconds
        adjusted_ipm = 60 / avg_interval
    else:
        adjusted_ipm = None  # Not enough data

    print(f"Filtered IPM: {filtered_ipm}")
    print(f"Adjusted IPM (based on intervals): {adjusted_ipm}")

    if len(valid_sp) < 2:
        # Not enough data to calculate heart rate
        return None, s.ppg

    # Step 7: Calculate heart rate from valid fiducial points
    Tpp = np.diff(valid_sp) / fs  # Convert peak-to-peak intervals to seconds
    heart_rate = 60 / Tpp  # Convert intervals to BPM

    # Step 8: Smooth the heart rate using moving average
    if len(heart_rate) > 5:  # Ensure enough data for smoothing
        heart_rate = moving_average(heart_rate, window_size=5)

    # Step 9: Compute the average heart rate
    avg_heart_rate = np.mean(heart_rate)

# hr.py

import pyPPG
from pyPPG import PPG
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.ppg_sqi as SQI
import numpy as np
from dotmap import DotMap
from scipy.signal import butter, filtfilt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def load_data_from_array(data, fs):
    """
    Mimic the load_data function to create a signal object from a numpy array.
    """
    s = DotMap()
    s.start_sig = 0
    s.end_sig = len(data)
    s.v = data
    s.fs = fs
    s.name = 'Real-time PPG Signal'
    return s



def high_pass_filter(data, cutoff, fs, order=4):
    """
    Apply a high-pass Butterworth filter to the data.
    
    :param data: The input signal.
    :param cutoff: The cutoff frequency of the high-pass filter.
    :param fs: The sampling frequency of the signal.
    :param order: The order of the filter.
    :return: The filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_heart_rate(data, fs, filtering=True, fL=0.5, fH=12.0, order=4, sqi_threshold=0.3):
    """
    Calculate the average heart rate from a PPG signal with SQI filtering.

    :param data: The PPG signal data as a numpy array.
    :param fs: Sampling frequency of the PPG signal.
    :param filtering: Whether to filter the PPG signal.
    :param fL: Lower cutoff frequency for filtering.
    :param fH: Upper cutoff frequency for filtering.
    :param order: Filter order.
    :param sqi_threshold: Minimum SQI value to consider a segment valid.
    :return: Average heart rate (BPM) and preprocessed PPG signal.
    """
    # Ensure data is a numpy array
    data = np.array(data)

    # Step 1: Load the PPG signal from array
    s = load_data_from_array(data, fs)

    # Step 2: Preprocess the PPG signal (filtering and derivatives)
    prep = PP.Preprocess(fL=fL, fH=fH, order=order,
                         sm_wins={'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10})
    s.ppg, s.vpg, s.apg, s.jpg = prep.get_signals(s=s)

    # Step 3: Create a PPG class
    s = PPG(s=s, check_ppg_len = False)


    # Step 4: Extract fiducial points
    fpex = FP.FpCollection(s=s)

    fiducials = fpex.get_fiducials(s=s)
    print(f"sp len : {len(fiducials.sp)}")

    total_seconds = len(data)/ 100
    print(f"Total second {total_seconds}")

    print( f"Raw IPM: {len(fiducials.sp)} * {int(60/total_seconds)}")


    # Step 5: Calculate SQI for the PPG signal
    sqi = SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fiducials.sp)

    # Step 6: Filter out low-quality segments based on SQI
    valid_indices = np.where(sqi > sqi_threshold)[0]  # Indices with SQI > threshold
    valid_sp = fiducials.sp.values[valid_indices]  # Filter systolic peaks with valid SQI


    # Calculate IPM
    total_seconds = len(data) / fs
    filtered_ipm = len(valid_sp) * (60 / total_seconds)

    # Adjust IPM based on average interval
    if len(valid_sp) > 1:
        avg_interval = np.mean(np.diff(fiducials.sp) / fs)  # Average interval in seconds
        adjusted_ipm = 60 / avg_interval
    else:
        adjusted_ipm = None  # Not enough data

    print(f"Filtered IPM: {filtered_ipm}")
    print(f"Adjusted IPM (based on intervals): {adjusted_ipm}")

    if len(valid_sp) < 2:
        # Not enough data to calculate heart rate
        return None, s.ppg

    # Step 7: Calculate heart rate from valid fiducial points
    Tpp = np.diff(valid_sp) / fs  # Convert peak-to-peak intervals to seconds
    heart_rate = 60 / Tpp  # Convert intervals to BPM

    # Step 8: Smooth the heart rate using moving average
    if len(heart_rate) > 5:  # Ensure enough data for smoothing
        heart_rate = moving_average(heart_rate, window_size=5)

    # Step 9: Compute the average heart rate
    avg_heart_rate = np.mean(heart_rate)

# hr.py

import pyPPG
from pyPPG import PPG
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.ppg_sqi as SQI
import numpy as np
from dotmap import DotMap
from scipy.signal import butter, filtfilt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def load_data_from_array(data, fs):
    """
    Mimic the load_data function to create a signal object from a numpy array.
    """
    s = DotMap()
    s.start_sig = 0
    s.end_sig = len(data)
    s.v = data
    s.fs = fs
    s.name = 'Real-time PPG Signal'
    return s



def high_pass_filter(data, cutoff, fs, order=4):
    """
    Apply a high-pass Butterworth filter to the data.
    
    :param data: The input signal.
    :param cutoff: The cutoff frequency of the high-pass filter.
    :param fs: The sampling frequency of the signal.
    :param order: The order of the filter.
    :return: The filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_heart_rate(data, fs, filtering=True, fL=0.5, fH=12.0, order=4, sqi_threshold=0.3):
    """
    Calculate the average heart rate from a PPG signal with SQI filtering.

    :param data: The PPG signal data as a numpy array.
    :param fs: Sampling frequency of the PPG signal.
    :param filtering: Whether to filter the PPG signal.
    :param fL: Lower cutoff frequency for filtering.
    :param fH: Upper cutoff frequency for filtering.
    :param order: Filter order.
    :param sqi_threshold: Minimum SQI value to consider a segment valid.
    :return: Average heart rate (BPM) and preprocessed PPG signal.
    """
    # Ensure data is a numpy array
    data = np.array(data)

    # Step 1: Load the PPG signal from array
    s = load_data_from_array(data, fs)

    # Step 2: Preprocess the PPG signal (filtering and derivatives)
    prep = PP.Preprocess(fL=fL, fH=fH, order=order,
                         sm_wins={'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10})
    s.ppg, s.vpg, s.apg, s.jpg = prep.get_signals(s=s)

    # Step 3: Create a PPG class
    s = PPG(s=s, check_ppg_len = False)


    # Step 4: Extract fiducial points
    fpex = FP.FpCollection(s=s)

    fiducials = fpex.get_fiducials(s=s)
    print(f"sp len : {len(fiducials.sp)}")

    total_seconds = len(data)/ 100
    print(f"Total second {total_seconds}")

    print( f"Raw IPM: {len(fiducials.sp)} * {int(60/total_seconds)}")


    # Step 5: Calculate SQI for the PPG signal
    sqi = SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fiducials.sp)

    # Step 6: Filter out low-quality segments based on SQI
    valid_indices = np.where(sqi > sqi_threshold)[0]  # Indices with SQI > threshold
    valid_sp = fiducials.sp.values[valid_indices]  # Filter systolic peaks with valid SQI


    # Calculate IPM
    total_seconds = len(data) / fs
    filtered_ipm = len(valid_sp) * (60 / total_seconds)

    # Adjust IPM based on average interval
    if len(valid_sp) > 1:
        avg_interval = np.mean(np.diff(fiducials.sp) / fs)  # Average interval in seconds
        adjusted_ipm = 60 / avg_interval
    else:
        adjusted_ipm = None  # Not enough data

    print(f"Filtered IPM: {filtered_ipm}")
    print(f"Adjusted IPM (based on intervals): {adjusted_ipm}")

    if len(valid_sp) < 2:
        # Not enough data to calculate heart rate
        return None, s.ppg

    # Step 7: Calculate heart rate from valid fiducial points
    Tpp = np.diff(valid_sp) / fs  # Convert peak-to-peak intervals to seconds
    heart_rate = 60 / Tpp  # Convert intervals to BPM

    # Step 8: Smooth the heart rate using moving average
    if len(heart_rate) > 5:  # Ensure enough data for smoothing
        heart_rate = moving_average(heart_rate, window_size=5)

    # Step 9: Compute the average heart rate
    avg_heart_rate = np.mean(heart_rate)

# hr.py

import pyPPG
from pyPPG import PPG
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.ppg_sqi as SQI
import numpy as np
from dotmap import DotMap
from scipy.signal import butter, filtfilt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def load_data_from_array(data, fs):
    """
    Mimic the load_data function to create a signal object from a numpy array.
    """
    s = DotMap()
    s.start_sig = 0
    s.end_sig = len(data)
    s.v = data
    s.fs = fs
    s.name = 'Real-time PPG Signal'
    return s



def high_pass_filter(data, cutoff, fs, order=4):
    """
    Apply a high-pass Butterworth filter to the data.
    
    :param data: The input signal.
    :param cutoff: The cutoff frequency of the high-pass filter.
    :param fs: The sampling frequency of the signal.
    :param order: The order of the filter.
    :return: The filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_heart_rate(data, fs, filtering=True, fL=0.5, fH=12.0, order=4, sqi_threshold=0.3):
    """
    Calculate the average heart rate from a PPG signal with SQI filtering.

    :param data: The PPG signal data as a numpy array.
    :param fs: Sampling frequency of the PPG signal.
    :param filtering: Whether to filter the PPG signal.
    :param fL: Lower cutoff frequency for filtering.
    :param fH: Upper cutoff frequency for filtering.
    :param order: Filter order.
    :param sqi_threshold: Minimum SQI value to consider a segment valid.
    :return: Average heart rate (BPM) and preprocessed PPG signal.
    """
    # Ensure data is a numpy array
    data = np.array(data)

    # Step 1: Load the PPG signal from array
    s = load_data_from_array(data, fs)

    # Step 2: Preprocess the PPG signal (filtering and derivatives)
    prep = PP.Preprocess(fL=fL, fH=fH, order=order,
                         sm_wins={'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10})
    s.ppg, s.vpg, s.apg, s.jpg = prep.get_signals(s=s)

    # Step 3: Create a PPG class
    s = PPG(s=s, check_ppg_len = False)


    # Step 4: Extract fiducial points
    fpex = FP.FpCollection(s=s)

    fiducials = fpex.get_fiducials(s=s)
    print(f"sp len : {len(fiducials.sp)}")

    total_seconds = len(data)/ 100
    print(f"Total second {total_seconds}")

    print( f"Raw IPM: {len(fiducials.sp)} * {int(60/total_seconds)}")


    # Step 5: Calculate SQI for the PPG signal
    sqi = SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fiducials.sp)

    # Step 6: Filter out low-quality segments based on SQI
    valid_indices = np.where(sqi > sqi_threshold)[0]  # Indices with SQI > threshold
    valid_sp = fiducials.sp.values[valid_indices]  # Filter systolic peaks with valid SQI


    # Calculate IPM
    total_seconds = len(data) / fs
    filtered_ipm = len(valid_sp) * (60 / total_seconds)

    # Adjust IPM based on average interval
    if len(valid_sp) > 1:
        avg_interval = np.mean(np.diff(fiducials.sp) / fs)  # Average interval in seconds
        adjusted_ipm = 60 / avg_interval
    else:
        adjusted_ipm = None  # Not enough data

    # print(f"Filtered IPM: {filtered_ipm}")
    # print(f"Adjusted IPM (based on intervals): {adjusted_ipm}")

    if len(valid_sp) < 2:
        # Not enough data to calculate heart rate
        return None, s.ppg

    # Step 7: Calculate heart rate from valid fiducial points
    Tpp = np.diff(valid_sp) / fs  # Convert peak-to-peak intervals to seconds
    heart_rate = 60 / Tpp  # Convert intervals to BPM

    # Step 8: Smooth the heart rate using moving average
    if len(heart_rate) > 5:  # Ensure enough data for smoothing
        smooth_heart_rate = moving_average(heart_rate, window_size=5)
    else:
        smooth_heart_rate = heart_rate  # Fallback to raw heart_rate

    hrstd = np.std(smooth_heart_rate)
    # print(f"HRSTD: {hrstd: .2f}")

    # Step 9: Compute the average heart rate
    if len(smooth_heart_rate) > 0:  # Check if there's data to calculate
        avg_heart_rate = np.mean(smooth_heart_rate)
    else:
        avg_heart_rate = None  # No data for average heart rate

    # Step 10: Calculate RMSSD
    if len(heart_rate) >= 2:  # At least two heart rate values are needed for RMSSD
        diff_rr_intervals = np.diff(heart_rate)
        squared_diffs = diff_rr_intervals ** 2
        mean_squared_diffs = np.mean(squared_diffs)
        rmssd = np.sqrt(mean_squared_diffs)
    else:
        rmssd = None  # Not enough data for RMSSD

    # print(f"RMSSD: {rmssd}")

    return avg_heart_rate, s.ppg, adjusted_ipm, rmssd, hrstd  # Return the average heart rate and the preprocessed PPG signal

