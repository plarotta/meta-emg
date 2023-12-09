from source.dataset import EMGDataset
from torch.utils.data import DataLoader

class EMGTask():
    def __init__(self, patient_path, condition, train_frac, bsize=32):
        self.train_data = EMGDataset(patient_path, condition, frac_data=train_frac)
        self.test_data = EMGDataset(patient_path, condition, frac_data=1-train_frac)
        self.trainloader = DataLoader(self.train_data, batch_size=bsize)
        self.testloader = DataLoader(self.test_data, batch_size=bsize)
        self.task_id = patient_path[-2:] + '_' + condition