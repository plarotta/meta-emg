from source.models import BasicCNN
import torch.optim as optim
from source.training_utils import maml, load_in_task_collection, get_save_dirs, get_baseline1, get_baseline2, process_logs, eval_trained_meta
from sklearn.model_selection import train_test_split
import numpy.random
from torch import manual_seed
import os

import torch
from matplotlib import pyplot as plt


def main():
    # SET SEEDS FOR REPRODUCIBILITY


    TC_PATH = r'C:\Users\plarotta\software\meta-emg\data\task_collections\patient_tc_train.json'
    N_VAL_TASKS = 5
    INNER_STEPS = 20
    INNER_LR = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb_logger = None
    # GET TEST-VAL SPLIT 
    task_colxn = load_in_task_collection(TC_PATH)
    train_colxn, val_clxn = train_test_split(task_colxn, test_size=N_VAL_TASKS)
    meta = BasicCNN()


    test_clxn = load_in_task_collection(r'C:\Users\plarotta\software\meta-emg\data\task_collections\patient_tc_test.json')
    base1_logs = get_baseline1(BasicCNN(), test_clxn, INNER_STEPS, INNER_LR, wandb_logger, device=device) # blank aka self
    base2_logs = get_baseline2(BasicCNN(), train_colxn, test_clxn, INNER_STEPS, INNER_LR,device=device, wandb=wandb_logger) # pre training aka fine-tuned
    
    meta.load_state_dict(torch.load(r'C:\Users\plarotta\software\meta-emg\data\expt_outputs\2023-12-21\16-49-55\models\epoch_0041_loss_0.7635\model_state_dict.pth'))
    m_logs = eval_trained_meta(meta, test_clxn, INNER_STEPS, INNER_LR, device=device)

    res_table, fig = process_logs(m_logs, base1_logs, base2_logs)
    res_table.to_csv(os.path.join(r'C:\Users\plarotta\software\meta-emg','res_table.csv'))
    plt.show()


if __name__ == '__main__':
    main()
