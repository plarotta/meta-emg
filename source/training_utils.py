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
                   baseline=0
                   ) -> dict:
    
    if store_grads:
        inner_optimizer = optim.Adam(model.parameters(), lr=inner_lr)
        # this wrapper is what allows us to store the inner loop gradients for the meta update
        with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
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
                    wandb.log({'loop':'fine-tuning','type': 'training', 'task_id': task.task_id, 't_loss': running_loss})
            
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
                    correct += torch.sum(torch.argmax(F.softmax(preds,dim=1),dim=1) == y_batch).item()
                    total_items += len(y_batch)

        val_loss = val_loss/(j+1)
        val_accuracy = correct/total_items
        if wandb:
            wandb.log({'loop':'fine-tuning','type': 'validation', 'task_id': task.task_id, 'v_loss': val_loss, 'v_accuracy': val_accuracy})

    else:
        model_copy = copy.deepcopy(model)
        inner_optimizer = optim.Adam(model_copy.parameters(), lr=inner_lr)
        loss_fct = nn.CrossEntropyLoss()
        training_losses = []

        # fine tune meta on current task
        for _ in range(inner_training_steps):
            running_loss = 0.0

            for i, (x_batch, y_batch) in enumerate(task.trainloader):
                inner_optimizer.zero_grad()
                loss = loss_fct(model_copy(x_batch.to(device)), 
                                y_batch.type(torch.LongTensor).to(device))
                running_loss += loss.item()
                loss.backward()
                inner_optimizer.step()
                
            running_loss = running_loss/(i+1)
            training_losses.append(running_loss)
            if wandb:
                wandb.log({'loop':'meta validation',
                           'type': 'training', 
                           'task_id': task.task_id, 
                           't_loss': running_loss,
                           'baseline': baseline})
        
        val_loss = 0.0
        correct = 0
        total_items = 0
        # get val loss for task
        with torch.no_grad():
            for j, (x_batch, y_batch) in enumerate(task.testloader):
                    preds = model_copy(x_batch.to(device))
                    loss = loss_fct(model_copy(x_batch.to(device)),
                                    y_batch.type(torch.LongTensor).to(device))
                    val_loss += loss.item()
                    correct += torch.sum(torch.argmax(F.softmax(preds,dim=1),dim=1) == y_batch).item()
                    total_items += len(y_batch)

        val_loss = val_loss/(j+1)
        val_accuracy = correct/total_items
        if wandb:
            wandb.log({'loop':'meta validation',
                       'type': 'validation', 
                       'task_id': task.task_id, 
                       'v_loss': val_loss, 
                       'v_accuracy': val_accuracy,
                       'baseline':baseline})

        print(f'task: {task.task_id}, val loss: {val_loss}, val accuracy: {val_accuracy}')
    
    return {'training_losses': training_losses, 'val_loss': val_loss, 'val_accuracy': val_accuracy}

def maml(meta_model: nn.Module, 
         training_tasks: list, 
         val_tasks: list,
         meta_optimizer: optim.Optimizer, 
         inner_training_steps: int, 
         meta_training_steps: int, 
         inner_lr: float,
         n_tasks=3,
         model_save_dir=None,
         wandb=None) -> dict:
    """
    Algorithm from https://arxiv.org/pdf/1703.03400v3.pdf (MAML for Few-Shot Supervised Learning)
    """
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    logger = {'train':{},'val':{}}
    for t in val_tasks:
        logger['val'][t.task_id] = [] 

    for epoch in tqdm(range(meta_training_steps)):  # Line 2 in the pseudocode

        tasks = sample_tasks(training_tasks, n_tasks) # Line 3 in the pseudocode
        meta_optimizer.zero_grad()
        for task in tasks:
            if task.task_id not in logger['train']:
                logger['train'][task.task_id] = []
            task_training_log = _fine_tune_model(meta_model,
                                               task,
                                               inner_training_steps,
                                               inner_lr,
                                               device=device,
                                               store_grads=True)
            logger['train'][task.task_id].append(task_training_log)
        meta_optimizer.step()  # Line 10 in the pseudocode

        [logger['val'][t.task_id].append(
            _fine_tune_model(meta_model, t, inner_training_steps, inner_lr, store_grads=False)) 
            for t in val_tasks]
        
        if model_save_dir:
            avg_val_loss = np.mean([logger['val'][t.task_id][-1]['val_loss'] for t in val_tasks])
            model_folder_name = f'epoch_{epoch:04d}_loss_{avg_val_loss:.4f}'
            os.makedirs(os.path.join(model_save_dir, model_folder_name))
            torch.save(meta_model.state_dict(), 
                       os.path.join(model_save_dir, model_folder_name, 'model_state_dict.pth'))
    
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

def load_in_task_collection(filepath):
    # Load task collection from a JSON file
    tc_list = _safe_json_load(filepath)
    curr_wd = os.getcwd()
    assert curr_wd[-8:] == 'meta-emg', "train.py must be run from the root of the meta-emg directory"
    root_dir = os.path.join(curr_wd,'data','collected_data')
    task_collection = [
        EMGTask(os.path.join(root_dir, d['session']), d['condition'], train_frac=0.25) 
        for d in tc_list if 'Augmen' not in d['session']]

    return(task_collection)

def get_save_dirs(outpit_root_dir: str) -> list:
    model_save_dir = os.path.join(outpit_root_dir, 'models')
    res_save_dir = os.path.join(outpit_root_dir, 'results')
    
    os.makedirs(model_save_dir)
    os.makedirs(res_save_dir)

    return(model_save_dir, res_save_dir)

def get_baseline1(blank_model: nn.Module, val_tasks: list, inner_steps: int, inner_lr: float, wandb=None):
    logger = {'train':{},'val':{}}
    for task in val_tasks:
        logger['train'][task.task_id] = []
        logger['val'][task.task_id] = []
    
    print('\nBASELINE 1: no meta training, no pre-training\n')
    [logger['val'][t.task_id].append(
            _fine_tune_model(blank_model, t, inner_steps, inner_lr, store_grads=False, wandb=wandb, baseline=1)) 
            for t in val_tasks]
    print('\n')
    return(logger)


def get_baseline2(blank_model: nn.Module, 
                  train_tasks: list[EMGTask], 
                  val_tasks: list, 
                  inner_steps: int, 
                  inner_lr: float, 
                  wandb=None,
                  device='cpu'):
    # generate big training dataset
    big_X = None
    big_Y = None
    for task in train_tasks:
        d = EMGDataset(task.session_path, task.condition, frac_data=1)
        
        if big_X is None:
            big_X = np.copy(d.emg_signals)
            big_Y = np.copy(d.labels)
            continue
        big_X = np.concatenate((big_X, d.emg_signals))
        big_Y = np.concatenate((big_Y, d.labels))
    
    mega_ds = EMGDataset(task.session_path, task.condition)
    mega_ds.emg_signals = big_X
    mega_ds.labels = big_Y

    trainloader = DataLoader(mega_ds, batch_size=32)

    optimizer = optim.Adam(blank_model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()
    print("Pre-training baseline 2")
    for epoch in tqdm(range(30)):
        running_loss = 0.0
        correct = 0
        tot = 0
        for i, (x_batch, y_batch) in enumerate(trainloader):
            optimizer.zero_grad()
            preds = blank_model(x_batch.to(device))
            loss = criterion(preds, 
                             y_batch.type(torch.LongTensor).to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            correct += torch.sum(torch.argmax(F.softmax(preds,dim=1),dim=1) == y_batch).item()
            tot += len(y_batch)
        running_loss = running_loss/(i+1)
        accuracy = correct/tot
        # print(f'epoch: {epoch} | training loss: {running_loss} | training accuracy: {accuracy}')


    logger = {'train':{},'val':{}}
    for task in val_tasks:
        logger['train'][task.task_id] = []
        logger['val'][task.task_id] = []

    print('\nBASELINE 2: no meta training, pre-trained on the entire train task collection\n')
    [logger['val'][t.task_id].append(
            _fine_tune_model(blank_model, t, inner_steps, inner_lr, store_grads=False, wandb=wandb, baseline=2)) 
            for t in val_tasks]
    
    return(logger)








