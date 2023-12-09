from torch.utils.data import Dataset
import numpy as np
import scipy.signal as sig
from pandas import read_csv


class EMGDataset(Dataset):
  'Creates an EMG dataset for PyTorch'
  def __init__(self,
               subject_file_path: str,
               condition: str,
               time_seq_len=25,
               filter=False,
               scaler=None,
               frac_data=1):

        self.raw_x, self.raw_y = self.extract_data(subject_file_path, condition, frac_data)

        self.emg_signals, self.labels = self.process_data(self.raw_x,
                                                          self.raw_y,
                                                          time_seq_len = time_seq_len,
                                                          filter = filter,
                                                          scaler = scaler)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        X = self.emg_signals[index,:]
        y = self.labels[index]

        return(X, y)

  def extract_data(self,
                   file_path: str,
                   condition: str,
                   frac_data: str
                   ) -> list[np.array, np.array]:

      df = read_csv(str(file_path + '/' + file_path[-2:] +'_'+ condition + '.csv'))
      df = df[["emg0","emg1","emg2","emg3","emg4","emg5","emg6","emg7","gt"]]
      raw_x = df.loc[:,df.columns != "gt"].to_numpy()[:int(len(df)*frac_data)]
      raw_y = df.loc[:,df.columns == "gt"].to_numpy().flatten()[:int(len(df)*frac_data)]
      return(raw_x, raw_y)

  def process_data(self,
                   x_raw: np.array,
                   y_raw: np.array,
                   time_seq_len=25,
                   filter=False,
                   scaler=None
                   ) -> list[np.array, np.array]:
      if filter:
          filter=sig.butter(N=1, fs =100, Wn = 30, btype='lowpass',output='sos')
          for lead in range(x_raw.shape[1]):
              x_raw[:,lead] = sig.sosfilt(filter, x_raw[:,lead])

      if scaler:
          raw_x = scaler.fit_transform(raw_x)
      split_arrays = np.split(x_raw[:x_raw.shape[0] // time_seq_len * time_seq_len], x_raw.shape[0] //time_seq_len)
      data_x = np.array(split_arrays, dtype = np.float64)
      data_y = np.split(y_raw[:y_raw.shape[0] // time_seq_len * time_seq_len], y_raw.shape[0] //time_seq_len)
      data_y = np.round(np.mean(data_y,axis=1))
      return(data_x, data_y)
