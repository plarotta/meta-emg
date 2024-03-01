from source.dataset import EMGDataset

from torch.utils.data import DataLoader
from pandas import read_csv
import matplotlib.pyplot as plt
import os


class EMGTask():
    def __init__(self, session_path, condition, bsize=32, time_series_len=25, stride=1, scale=0, rorcr_size=-1):
        self.session_path = session_path
        self.condition = condition
        self.rorcr_split_index = self.get_rorcr_idx(session_path, condition)
        self.train_data = EMGDataset(session_path, 
                                     condition, 
                                     rorcr_idx=self.rorcr_split_index, 
                                     train=True, 
                                     time_seq_len=time_series_len, 
                                     stride=stride, 
                                     scale=scale,
                                     rorcr_sample_size=rorcr_size)
        self.test_data = EMGDataset(session_path, 
                                    condition, 
                                    rorcr_idx=self.rorcr_split_index, 
                                    train=False, 
                                    time_seq_len=time_series_len, 
                                    stride=stride, 
                                    scale=scale,
                                    scaler=self.train_data.scaler if scale == 1 else None,
                                    rorcr_sample_size=rorcr_size) # use stats from train set
        self.trainloader = DataLoader(self.train_data, batch_size=bsize)
        self.testloader = DataLoader(self.test_data, batch_size=bsize)
        self.task_id = session_path[-2:] + '_' + condition

    
    def get_rorcr_idx(self, session_path, condition):
        '''method that gets the split index for a recording such that
        the train split is the first rorcr 
        '''
        
        gts = read_csv(str(session_path + '/' + session_path[-2:] +'_'+ condition + '.csv'))
        gts = gts[["gt"]].to_numpy()

        # rorcr is defined as these transitions in the correct order
        transitions = [1,-1,2,-2,1]
        t_idx = 0

        for idx in range(len(gts)-1):
            if t_idx > 4:
                break
            curr=gts[idx]
            next=gts[idx+1]

            # transition logging
            if curr == next:
                continue

            assert next-curr == transitions[t_idx], f'Problem getting RORCR idx. Inspect {session_path}/{session_path[-2:]}_{condition}'
            
            t_idx+=1
        return(idx)

if __name__ =='__main__':
    a = EMGTask(os.path.join(r'C:\Users\plarotta\software\meta-emg\data\collected_data', '2023_09_27_p4'), '111', 
    ) 

    # plt.plot(a.train_data.raw_x)
    plt.plot(a.train_data.raw_x)
    plt.plot(a.train_data.raw_y*300,'k')

    plt.show()