from torch.utils.data import Dataset
import numpy as np
import scipy.signal as sig
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class EMGDataset(Dataset):
    '''Creates an EMG dataset for PyTorch'''
    def __init__(self,
                subject_file_path: str,
                condition: str,
                time_seq_len=25,
                scale=0, #0 for no scale, 1 for standardization, 2 for -1 to 1 scaling
                scaler=None,
                rorcr_idx=-1,
                train=True,
                stride=1):
        self.train = train
        self.scale = scale
        self.time_seq_len = time_seq_len
        self.stride = stride
        self.scaler = scaler

        self.raw_x, self.raw_y = self.extract_data(subject_file_path, condition, rorcr_idx)

        self.emg_signals, self.labels = self.process_data(self.raw_x,
                                                          self.raw_y)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        X = self.emg_signals[index,:]
        y = self.labels[index]

        return(X, y)

    def extract_data(self,
                    file_path: str,
                    condition: str,
                    rorcr_idx: str
                    ) -> list[np.array, np.array]:

        df = read_csv(str(file_path + '/' + file_path[-2:] +'_'+ condition + '.csv'))
        df = df[["emg0","emg1","emg2","emg3","emg4","emg5","emg6","emg7","gt"]]
        if rorcr_idx == -1:
            rorcr_idx = len(df)

        if self.train is True:
            raw_x = df.loc[:,df.columns != "gt"].to_numpy()[:rorcr_idx]
            raw_y = df.loc[:,df.columns == "gt"].to_numpy().flatten()[:rorcr_idx]
        else:
            raw_x = df.loc[:,df.columns != "gt"].to_numpy()[rorcr_idx:]
            raw_y = df.loc[:,df.columns == "gt"].to_numpy().flatten()[rorcr_idx:]

        return(raw_x, raw_y)


    def process_data(self,
                    x_raw: np.array,
                    y_raw: np.array,
                    ) -> list[np.array, np.array]:
        
        np.clip(x_raw, a_max=1000, a_min=0)
        
        if self.scale == 1:
            if self.train is True:
                self.scaler = StandardScaler()
                self.scaler.fit(x_raw)

            # if it's the test split, scaler is already fit to training data
            x_raw = self.scaler.transform(x_raw)
  
        elif self.scale == 2:
            scaler = MinMaxScaler()
            x_raw = scaler.fit_transform(x_raw)

        chunk_head = 0
        chunk_tail = self.time_seq_len
        
        signal_end = len(x_raw)
        x_holder = np.zeros(((signal_end-self.time_seq_len)//self.stride+1,  self.time_seq_len, 8))
        y_holder = np.zeros(((signal_end-self.time_seq_len)//self.stride+1,  self.time_seq_len))
        n = 0

        while chunk_tail < signal_end:
            x_chunk = x_raw[chunk_head:chunk_tail]
            y_chunk = y_raw[chunk_head:chunk_tail]

            x_holder[n, :, :] = x_chunk
            y_holder[n, :] = y_chunk

            chunk_head+=self.stride
            chunk_tail+=self.stride 
            n+=1

        y_holder = np.rint(np.mean(y_holder, axis=1))

        return(x_holder, y_holder)
    

if __name__ == '__main__':
    a = EMGDataset('/Users/plarotta/software/meta-emg/data/collected_data/2023_03_14_p4', '111', scale=2)
    print(a.emg_signals.shape)







