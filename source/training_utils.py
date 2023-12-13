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
import time
import copy
import warnings

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


def _fine_tune_meta(model: nn.Module, 
                   task: EMGTask, 
                   inner_training_steps: int, 
                   inner_lr: float, 
                   device='cpu', 
                   store_grads=False
                   ) -> dict:
    
    if store_grads:
        inner_optimizer = optim.SGD(model.parameters(), lr=inner_lr)
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

    else:
        model_copy = copy.deepcopy(model)
        inner_optimizer = optim.SGD(model_copy.parameters(), lr=inner_lr)
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
         model_save_dir=None) -> dict:
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
                logger['val'][task.task_id] = []
            task_training_log = _fine_tune_meta(meta_model,
                                               task,
                                               inner_training_steps,
                                               inner_lr,
                                               device=device,
                                               store_grads=True)
            logger['train'][task.task_id].append(task_training_log)
        meta_optimizer.step()  # Line 10 in the pseudocode

        [logger['val'][t.task_id].append(
            _fine_tune_meta(meta_model, t, inner_training_steps, inner_lr, store_grads=False)) 
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

