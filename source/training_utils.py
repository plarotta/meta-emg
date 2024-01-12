import numpy as np
import higher
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
from source.dataset import EMGDataset
from source.task import EMGTask
import json
import os
import copy
import warnings
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# hide warning from loss.backward(retain_graph=True)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd")

def sample_tasks(task_distribution: list, 
                 n_tasks: int
                 ) -> EMGDataset:
    # grab n_tasks number of samples from task_distribution
    out = np.take(task_distribution,
            indices=np.random.choice(
                list(range(len(task_distribution))),
                size=n_tasks,
                replace = False))
    return(out)


def _fine_tune_model(model: nn.Module, 
                    task: EMGTask, 
                    inner_training_steps: int, 
                    inner_lr: float, 
                    device='cpu', 
                    store_grads=False,
                    wandb=None,
                    baseline=None
                    ) -> dict:
    model = model.to(device)

    inner_optimizer = optim.Adam(model.parameters(), lr=inner_lr)
    # this wrapper is what allows us to store the inner loop gradients for the meta update
    with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False, track_higher_grads=store_grads) as (fmodel, diffopt):
        training_losses = []

        # fine tune meta on current task
        for _ in range(inner_training_steps):
            running_loss = 0.0

            for i, (x_batch, y_batch) in enumerate(task.trainloader):
                loss = F.cross_entropy(fmodel.forward(x_batch.to(device)), 
                                    y_batch.type(torch.LongTensor).to(device))
                diffopt.step(loss)
                running_loss += loss.item()
            running_loss = running_loss/(i+1)
            training_losses.append(running_loss)
            if wandb:
                wandb.log({'meta/training/training_loss': running_loss})
        
        val_loss = 0.0
        correct = 0
        total_items = 0
        # get val loss for task, and send grad(theta_prime,theta) back 
        for j, (x_batch, y_batch) in enumerate(task.testloader):
                preds = fmodel.forward(x_batch.to(device))
                loss = F.cross_entropy(preds, 
                                        y_batch.type(torch.LongTensor).to(device))
                loss.backward(create_graph=True,)
                val_loss += loss.item()
                correct += torch.sum(torch.argmax(F.softmax(preds,dim=1),dim=1) == y_batch.to(device)).item()
                total_items += len(y_batch)

    val_loss = val_loss/(j+1)
    val_accuracy = correct/total_items
    if wandb:
        if store_grads:
            wandb.log({'meta/training/val_loss': val_loss,
                    'meta/training/val_acc': val_accuracy})
        elif baseline is not None:
                wandb.log({
                    f'baseline{baseline}/fine_tuning_training_loss': running_loss})
        else:
            wandb.log({
                'meta/validation/fine_tuning_training_loss':running_loss
            })
    return {'training_losses': training_losses, 'val_loss': val_loss, 'val_accuracy': val_accuracy}

def maml(meta_model: nn.Module, 
         training_tasks: list[EMGTask], 
         val_tasks: list[EMGTask],
         test_tasks: list[EMGTask],
         meta_optimizer: optim.Optimizer, 
         inner_training_steps: int, 
         meta_training_steps: int, 
         inner_lr: float,
         n_tasks=3,
         model_save_dir=None,
         wandb=None,
         test=True) -> dict:
    """
    Algorithm from https://arxiv.org/pdf/1703.03400v3.pdf (MAML for Few-Shot Supervised Learning)
    """
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger = {'train':{},'val':{}, 'test':{}}
    for t in val_tasks:
        logger['val'][t.task_id] = [] 

    for t in test_tasks:
        logger['test'][t.task_id] = []
    
    def coll_fn(batch):
        return(batch)
    
    trainloader = DataLoader(training_tasks, batch_size=n_tasks, shuffle=True, collate_fn=coll_fn)
    sched = optim.lr_scheduler.StepLR(meta_optimizer, 15, gamma=.75, verbose=True)

    for epoch in range(meta_training_steps):  # Line 2 in the pseudocode
        print(f'Meta epoch # {epoch}...')
        for batch_idx, tasks in enumerate(tqdm(trainloader)):
            meta_optimizer.zero_grad()
            for task in tasks:
                if task.task_id not in logger['train']:
                    logger['train'][task.task_id] = []
                task_training_log = _fine_tune_model(meta_model,
                                                task,
                                                inner_training_steps,
                                                inner_lr,
                                                device=device,
                                                store_grads=True,
                                                wandb=wandb)
                logger['train'][task.task_id].append(task_training_log)
            meta_optimizer.step()  # Line 10 in the pseudocode

        [logger['val'][t.task_id].append(
            _fine_tune_model(meta_model, t, inner_training_steps, inner_lr, store_grads=False)) 
            for t in val_tasks]
        
        avg_val_loss = np.mean([logger['val'][t.task_id][-1]['val_loss'] for t in val_tasks])
        avg_val_acc = np.mean([logger['val'][t.task_id][-1]['val_accuracy'] for t in val_tasks])
        avg_train_loss = np.mean([logger['train'][t.task_id][-1]['val_loss'] for t in training_tasks])
        avg_train_acc = np.mean([logger['train'][t.task_id][-1]['val_accuracy'] for t in training_tasks])
        
        if wandb:
            wandb.log({
                'meta/validation/avg_val_loss': avg_val_loss,
                'meta/validation/avg_val_acc': avg_val_acc,
                'meta/validation/epoch': epoch,
                'meta/training/avg_train_loss': avg_train_loss,
                'meta/training/avg_train_acc': avg_train_acc,
            })
        
        if model_save_dir:
            model_folder_name = f'epoch_{epoch:04d}_loss_{avg_val_loss:.4f}'
            os.makedirs(os.path.join(model_save_dir, model_folder_name))
            torch.save(meta_model.state_dict(), 
                    os.path.join(model_save_dir, model_folder_name, 'model_state_dict.pth'))

        print(f'average val accuracy: {avg_val_acc}')
        sched.step()

    # TODO: log these in wandb as well
    if test:
        [logger['test'][t.task_id].append(
            _fine_tune_model(meta_model, t, inner_training_steps, inner_lr, store_grads=False)) 
            for t in test_tasks]
    
    return logger

def _safe_json_load(filepath):
    try:
        with open(filepath, 'r') as json_file:
            task_collection = json.load(json_file)
            return(task_collection)
    except FileNotFoundError:
        print(f"File {filepath} not found. No task collection loaded.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file {filepath}. No task collection loaded.")

def load_in_task_collection(filepath, batch_size=32, time_seq=25, stride=1):
    # Load task collection from a JSON file
    tc_list = _safe_json_load(filepath)
    curr_wd = os.getcwd()
    assert curr_wd[-8:] == 'meta-emg', "train.py must be run from the root of the meta-emg directory"
    root_dir = os.path.join(curr_wd,'data','collected_data')
    task_collection = [
        EMGTask(os.path.join(root_dir, d['session']), d['condition'], 
                bsize=batch_size, 
                time_series_len=time_seq, 
                stride=stride
                ) 
        for d in tc_list if 'Augmen' not in d['session']]

    return(task_collection)

def get_save_dirs(outpit_root_dir: str) -> list:
    model_save_dir = os.path.join(outpit_root_dir, 'models')
    res_save_dir = os.path.join(outpit_root_dir, 'results')
    
    os.makedirs(model_save_dir)
    os.makedirs(res_save_dir)

    return(model_save_dir, res_save_dir)

def eval_trained_meta(model, test_tasks, inner_steps, inner_lr, wandb=None,device=None):
    logger = {'test':{}}
    for task in test_tasks:
        logger['test'][task.task_id] = []

    print('\nevaluating trained meta model...\n')
    [logger['test'][t.task_id].append(
            _fine_tune_model(model, t, inner_steps, inner_lr, store_grads=False, wandb=wandb, device=device)) 
            for t in test_tasks]
    print('\n')
    return(logger)

def get_baseline1(blank_model: nn.Module, 
                  test_tasks: list[EMGTask], 
                  inner_steps: int, 
                  inner_lr: float, 
                  wandb=None, 
                  device='cpu'):
    logger = {'test':{}}
    for task in test_tasks:
        logger['test'][task.task_id] = []
    
    print('\nBASELINE 1: no meta training, no pre-training\n')
    [logger['test'][t.task_id].append(
            _fine_tune_model(blank_model, t, inner_steps, inner_lr, store_grads=False, wandb=wandb, baseline=1, device=device)) 
            for t in test_tasks]
    print('\n')
    return(logger)


def get_baseline2(blank_model: nn.Module, 
                  train_tasks: list[EMGTask], 
                  test_tasks: list, 
                  inner_steps: int, 
                  inner_lr: float, 
                  wandb=None,
                  device='cpu',
                  stride=1,
                  time_seq_len=25,
                  batch_size=32):
    # generate big training dataset
    big_X = None
    big_Y = None
    for task in train_tasks:
        d = EMGDataset(task.session_path, task.condition, time_seq_len=time_seq_len, stride=stride)
        
        if big_X is None:
            big_X = np.copy(d.emg_signals)
            big_Y = np.copy(d.labels)
            continue
        big_X = np.concatenate((big_X, d.emg_signals))
        big_Y = np.concatenate((big_Y, d.labels))
    
    mega_ds = EMGDataset(task.session_path, task.condition) # doesnt matter
    mega_ds.emg_signals = big_X
    mega_ds.labels = big_Y

    trainloader = DataLoader(mega_ds, batch_size=batch_size)
    blank_model = blank_model.to(device)

    optimizer = optim.Adam(blank_model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()
    print("Pre-training baseline 2")
    for epoch in tqdm(range(50)):
        running_loss = 0.0
        correct = 0
        tot = 0
        for i, (x_batch, y_batch) in enumerate(trainloader):
            # print(x_batch.shape)
            optimizer.zero_grad()
            preds = blank_model(x_batch.to(device))
            loss = criterion(preds, 
                             y_batch.type(torch.LongTensor).to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            correct += torch.sum(torch.argmax(F.softmax(preds,dim=1),dim=1) == y_batch.to(device)).item()
            tot += len(y_batch)
        running_loss = running_loss/(i+1)
        accuracy = correct/tot
        if wandb:
            wandb.log({
                f'baseline2/pretraining_loss': running_loss,
                f'baseline2/pretraining_acc': accuracy,
                f'baseline2/pretraining_epoch': epoch,
            })


    logger = {'test':{}}
    for task in test_tasks:
        logger['test'][task.task_id] = []

    print('\nBASELINE 2: no meta training, pre-trained on the entire train task collection\n')
    [logger['test'][t.task_id].append(
            _fine_tune_model(blank_model, t, inner_steps, inner_lr, store_grads=False, wandb=wandb, baseline=2,device=device)) 
            for t in test_tasks]
    
    return(logger)

def process_logs(meta_log, b1_log, b2_log):

    # TODO: reduce these boys into a single for loop
    meta_accs = []
    meta_labs = []
    for t in meta_log['test']:
        meta_accs.append(meta_log['test'][t][-1]['val_accuracy'])
        meta_labs.append(t)

    
    b1_accs = []
    b1_labs = []
    for t in b1_log['test']:
        b1_accs.append(b1_log['test'][t][-1]['val_accuracy'])
        b1_labs.append(t)

    b2_accs = []
    b2_labs = []
    for t in b2_log['test']:
        b2_accs.append(b2_log['test'][t][-1]['val_accuracy'])
        b2_labs.append(t)

    assert b2_labs == b1_labs, 'task ordering issue in baseline plots'
    assert meta_labs == b2_labs, 'task ordering issue in baseline plots'

    # Calculate averages for each model
    model1_avg = np.mean(b1_accs)
    model2_avg = np.mean(b2_accs)
    model3_avg = np.mean(meta_accs)

    res_table = pd.DataFrame([[*b1_accs,model1_avg],[*b2_accs,model2_avg],[*meta_accs,model3_avg]], 
                       columns=[*meta_labs,'avg'],
                       index=pd.Index(['Baseline 1: no pre-training, no meta training','Baseline 2: pre-trained model','M-EMG',]))
    print(res_table)

    # Create a figure for the table
    fig, ax = plt.subplots(1, figsize=(20, 10))

    
    # Number of tasks
    num_tasks = len(meta_labs)

    # Set the bar width
    bar_width = 0.30

    # Set the positions of the bars on the x-axis
    r1 = np.arange(num_tasks)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Plotting the grouped bar chart
    ax.bar(r1, b1_accs, width=bar_width, label='Baseline 1: no pre-training, no meta training')
    ax.bar(r2, b2_accs, width=bar_width, label='Baseline 2: pre-trained model')
    ax.bar(r3, meta_accs, width=bar_width, label='M-EMG')

    # Adding labels and title
    ax.set_xlabel('Tasks')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performances on Different Tasks')
    ax.set_xticks([r + bar_width for r in range(num_tasks)], meta_labs)
    ax.set_xticklabels(meta_labs, rotation=45, ha='right')
    ax.legend()

    return(res_table, fig)
    






