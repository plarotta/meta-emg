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
                stride=1,
                rorcr_sample_size=1):
        self.rorcr_idx = rorcr_idx
        self.train = train
        self.scale = scale
        self.time_seq_len = time_seq_len
        self.stride = stride
        self.scaler = scaler

        self.raw_x, self.raw_y = self.extract_data(subject_file_path, condition, rorcr_idx, rorcr_sample_size)

        self.emg_signals, self.labels = self.process_data(self.raw_x,
                                                          self.raw_y)

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.labels)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        X = self.emg_signals[index,:]
        y = self.labels[index]
        return(X, y)

    def extract_data(self,
                     file_path: str,
                     condition: str,
                     rorcr_idx: str,
                     rorcr_sample_size=1
                     ) -> list[np.array, np.array]:

        df = read_csv(str(file_path + '/' + file_path[-2:] +'_'+ condition + '.csv'))
        df = df[["emg0","emg1","emg2","emg3","emg4","emg5","emg6","emg7","gt"]]
        if rorcr_idx == -1:
            rorcr_idx = len(df)

        if self.train is True:
            raw_x = df.loc[:,df.columns != "gt"].to_numpy()[:rorcr_idx]
            raw_y = df.loc[:,df.columns == "gt"].to_numpy().flatten()[:rorcr_idx]
            raw_x,raw_y = self.sample_from_rorcr(raw_x, raw_y, rorcr_sample_size)
        else:
            raw_x = df.loc[:,df.columns != "gt"].to_numpy()[rorcr_idx:]
            raw_y = df.loc[:,df.columns == "gt"].to_numpy().flatten()[rorcr_idx:]

        return(raw_x, raw_y)

    def sample_from_rorcr(self, x, y, sample_size=1):
        if sample_size >= 1:
            # print('FULL RORCR')
            return(x,y)
        else:
            # _,ax = plt.subplots(2)
            # ax[0].plot(x)
            # ax[0].plot(y*100)
            # ax[0].set_xlim(0,self.rorcr_idx)
            
            output_x = np.zeros((int(self.rorcr_idx*sample_size), 8))
            output_y = np.zeros((int(self.rorcr_idx*sample_size)))
            labs = [0,1,0,2,0]
            lab_ptr = 0
            output_idx = 0
            n = 0
            for data_idx in range(len(y)):
                if y[data_idx] == labs[lab_ptr]:
                    output_y[output_idx] = y[data_idx]
                    output_x[output_idx,:] = x[data_idx]
                    output_idx += 1
                    n +=1
                if n >= int(self.rorcr_idx*sample_size/5):
                    # print(int(self.rorcr_idx*sample_size)/5)
                    lab_ptr +=1
                    n = 0
                if lab_ptr >= 5:
                    break
            # ax[1].plot(output_x)
            # ax[1].plot(output_y*100)
            # ax[1].set_xlim(0,self.rorcr_idx)
            # plt.show()
            return(output_x,output_y)


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
        y_holder = y_holder[:,-1]

        return(x_holder, y_holder)
    

if __name__ == '__main__':
    a = EMGDataset('/Users/plarotta/software/meta-emg/data/collected_data/2023_03_14_p4', '111', scale=2, rorcr_sample_size=1, rorcr_idx=2745)







