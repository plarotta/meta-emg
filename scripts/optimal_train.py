import optuna
from source.models import BasicCNN
import torch.optim as optim
from source.training_utils import maml, load_in_task_collection, get_save_dirs, get_baseline1, get_baseline2, process_logs
from sklearn.model_selection import train_test_split
import numpy.random
from torch import manual_seed
import pickle
import os
import wandb
import torch
import numpy as np


def objective(trial: optuna.Trial):
    numpy.random.seed(0)
    manual_seed(0)

    INNER_LR = trial.suggest_float('inner_lr', low=1e-5,high=1e-2, log=True)
    OUTER_LR = trial.suggest_float('outer_lr', low=1e-5,high=1e-1, log=True)
    META_STEPS = 5
    INNER_STEPS = trial.suggest_int('inner_steps', low=10, high=25)

    N_VAL_TASKS = 5
    N_TRAIN_TASKS = 3

    TC_PATH = r'C:\Users\plarotta\software\meta-emg\data\task_collections\patient_tc_train.json'
    OUT_ROOT = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task_colxn = load_in_task_collection(TC_PATH)
    train_colxn, val_clxn = train_test_split(task_colxn, test_size=N_VAL_TASKS)
    test_clxn = load_in_task_collection(r'C:\Users\plarotta\software\meta-emg\data\task_collections\patient_tc_test.json')
    fc_features = trial.suggest_int('fc_features', low=16, high=400)
    meta_model = BasicCNN(fc_features)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    meta_optimizer = getattr(optim, optimizer_name)(meta_model.parameters(), lr=OUTER_LR)

    maml_logs = maml(meta_model, 
                    train_colxn,
                    val_clxn,
                    test_clxn,
                    meta_optimizer, 
                    INNER_STEPS, 
                    META_STEPS, 
                    INNER_LR, 
                    n_tasks=N_TRAIN_TASKS,
                    test=False)
    
    avg_val_acc = np.mean([maml_logs['val'][t.task_id][-1]['val_accuracy'] for t in val_clxn])
    
    return avg_val_acc

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)