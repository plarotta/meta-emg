from source.dataset import EMGDataset
from torch.utils.data import DataLoader

class EMGTask():
    def __init__(self, session_path, condition, train_frac, bsize=32):
        self.session_path = session_path
        self.condition = condition
        self.train_data = EMGDataset(session_path, condition, frac_data=train_frac)
        self.test_data = EMGDataset(session_path, condition, frac_data=1-train_frac)
        self.trainloader = DataLoader(self.train_data, batch_size=bsize)
        self.testloader = DataLoader(self.test_data, batch_size=bsize)
        self.task_id = session_path[-2:] + '_' + condition