import csv, os
import numpy as np
import pandas as pd
from biosppy.signals import tools as st
from biosppy import plotting, utils
import zipfile
import matplotlib.pyplot as plt

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

  '''
  arguments:
      data: sensor data to be filtered
      low: lower frequency threshold
      high: upper frequency threshold

  returns: filtered dataset with every patient inside
  '''

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

'''
with zipfile.ZipFile('C:/users/naten/documents/coding/merlin/blink/Datasets/Original/blink/EEG-IO.zip', 'r') as zip_ref:
    zip_ref.extractall('C:/users/naten/documents/coding/merlin/blink/unzipped')
'''

path = 'C:/users/naten/documents/coding/merlin/blink/unzipped/'
path2 = 'C:/users/naten/documents/coding/merlin/blink/filtered/'

titles = [f for f in os.listdir(path) if '_data' in f]
data_stim_files = [path+f for f in os.listdir(path) if '_data' in f]
data_labels_files = [data.replace('_data', '_labels') for data in data_stim_files]

data_stim = [np.loadtxt(open(item, "rb"), delimiter=';', skiprows=1, usecols=(1,2)) for item in data_stim_files]
data_labels = [np.loadtxt(lbl, delimiter = ',', skiprows = 3) for lbl in data_labels_files]


filtered = get_filtered(data_stim, 250, 4, 44)

for i in range(len(filtered)):
    np.savetxt(path2+titles[i], filtered[i], delimiter=',')
