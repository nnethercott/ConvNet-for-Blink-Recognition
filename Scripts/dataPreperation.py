import numpy as np
import os, csv
import matplotlib.pyplot as plt
from biosppy.signals import tools as st
from biosppy import plotting, utils

#could just import the module Awni made earlier 

def biolowhigh(signal, sampling_rate, low, high, mode='includehigh'):
  if mode == 'includehigh': #think i included this for testing purposes
    b, a = st.get_filter(ftype='butter',
                          band='highpass',
                          order=8,
                          frequency=low,
                          sampling_rate=sampling_rate)

    aux, _ = st._filter_signal(b, a, signal=signal, check_phase=True, axis=0)

  if mode == 'nohigh':
    aux = signal

  # low pass filter
  b, a = st.get_filter(ftype='butter',
                        band='lowpass',
                        order=16,
                        frequency=high,
                        sampling_rate=sampling_rate)

  filtered, _ = st._filter_signal(b, a, signal=aux, check_phase=True, axis=0)
  return filtered

def get_filtered(data, sampling_rate, low, high):

  """
  arguments:
      data: sensor data to be filtered
      low: lower frequency threshold
      high: upper frequency threshold

  returns: filtered dataset with every patient inside
  """

  data = np.array(data)
  newdata = data
  #now iterate through each entry
  for i in range(len(data)):
    ndim = data[i].shape[1]
    for j in range(ndim):
      temp=newdata[i]
      senstemp = temp[:,j]
      senstemp = biolowhigh(senstemp, sampling_rate, low, high)
      temp[:,j] = senstemp
    newdata[i] = temp

  return newdata

def create_windows(data, window_size, inter_window_interval):
    """
    Creates overlapping windows of EEG data from a single recording.

    Arguments:
        data: array of timeseries data
        window_size(int): desired size of each window in number of samples
        inter_window_interval(int): interval between each window in number of samples

    Returns:
        numpy array object with the windows along its first dimension
    """

    #Calculate the number of valid windows of the specified size which can be retrieved from data
    #Explanation: the max overflow possible is window_size, so we check how many
    num_windows = 1 + (len(data) - window_size) // inter_window_interval

    windows = []
    for i in range(num_windows):
        windows.append(data[i*inter_window_interval:i*inter_window_interval + window_size])

    return np.array(windows)

def labels_from_timestamps(timestamps, sampling_rate, length):
    """takes an array containing timestamps (as floats) and
    returns a labels array of size 'length' where each index
    corresponding to a timestamp via the 'samplingRate'.

    Arguments:
        timestamps: an array containing the timestamps for each event (units must match sampling_rate).
        sampling_rate: the sampling rate of the EEG data.
        length: the number of samples of the corresponing EEG run.

    Returns:
        an integer array of size 'length' with a '1' at each
        time index where a corresponding timestamp exists, and a '0' otherwise
    """

    #create labels array template
    labels = np.zeros(length)

    #calculate corresponding index for each timestamp
    labelIndices = timestamps * sampling_rate
    labelIndices = labelIndices.astype(int)

    #flag label at each index where a corresponding timestamp exists
    labels[labelIndices] = 1
    labels = labels.astype(int)

    return labels

def label_windows(labels, window_size, inter_window_interval, label_method):
  windows = create_windows(labels, window_size, inter_window_interval)

  if label_method == "containment":
    window_labels = [int(1 in window) for window in windows]
  elif label_method == "count":
    window_labels = [np.sum(window) for window in windows]
  elif label_method == "mode":
    #choose the most common occurence in the window and default to 0 if multiple exist
    window_labels = [stats.mode(window)[0][0] for window in windows]

  return window_labels

def generate_samples(data, labels, window_size, window_step, sampling_rate):
  """
  arguments:
      data: array of sensor readings
      labels: timestamps associated with blink
      window_size: desired size of input vector for ML model
      window_step: size of index increment

  returns:
      windowed data and window labels
  """
  data_windows = create_windows(data, window_size, window_step)
  label_list = labels_from_timestamps(labels, sampling_rate, len(data))
  lbl_windows = label_windows(label_list, window_size, window_step, "containment")
  return data_windows, lbl_windows

def data_prep(dataset, labels, sampling_rate, window_size, window_step, sensor):
  """
  returns:
      flattened list of normalized windowed data with associated window labels from a single subject
  """
  list_of_windows = []
  list_of_labels = []

  number_of_subjects = len(dataset) #change this to first value in shape array since one dimensional might just give a shit ton of stuff
  placeholder = np.array(dataset)

  for i in range(number_of_subjects): #why not use the get_filter function? something to do with having to already iterate through subjects
    subject_data = placeholder[i]
    subject_labels = labels[i]
    number_of_channels = placeholder[i].shape[1]

    filtered_channel = biolowhigh(subject_data[:,sensor-1], sampling_rate, 4, 44)  #standard
    filtered_windows, filtered_window_labels = generate_samples(filtered_channel, subject_labels, window_size, window_step, sampling_rate)

    for k in range(len(filtered_window_labels)):
      list_of_windows.append(filtered_windows[k]) #microvolts to volts
      list_of_labels.append(filtered_window_labels[k])


  return np.array(list_of_windows), np.array(list_of_labels)
