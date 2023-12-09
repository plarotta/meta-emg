import numpy as np
import higher
import torch.optim as optim
import torch.nn.functional as F
import torch
import tqdm
from dataset import EMGDataset

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


def fine_tune_meta(model, task, inner_training_steps, alpha, meta_opt, device='cpu'):

    inner_optimizer = optim.SGD(model.parameters(), lr=1e-4)
    print(f'\nfine tuning meta on task: {task.task_id}...')

    # this wrapper is what allows us to store the inner loop gradients for the meta update
    with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
        training_losses = []

        # fine tune meta on current task
        for epoch in range(inner_training_steps):
            running_loss = 0.0

            for i, (x_batch, y_batch) in enumerate(task.trainloader):
                loss = F.cross_entropy(fmodel.forward(x_batch.to(device)), y_batch.type(torch.LongTensor).to(device))
                diffopt.step(loss)
                running_loss += loss.item()
            running_loss = running_loss/(i+1)
            training_losses.append(running_loss)
        val_loss = 0.0

        # get val loss for task, and send grad(theta_prime,theta) back 
        for j, (x_batch, y_batch) in enumerate(task.testloader):
                loss = F.cross_entropy(fmodel.forward(x_batch.to(device)), y_batch.type(torch.LongTensor).to(device))
                loss.backward(create_graph=True)
                val_loss += loss.item()

    val_loss = val_loss/(j+1)
    print(f'val loss: {val_loss}')
    return {'training_losses': training_losses, 'val_loss': val_loss}


def maml(meta_model, 
         task_dist, 
         meta_optimizer, 
         inner_training_steps, 
         meta_training_steps, 
         alpha, 
         n_tasks=3, 
         device='cpu') -> dict:
    """
    Algorithm from https://arxiv.org/pdf/1703.03400v3.pdf (MAML for Few-Shot Supervised Learning)
    """
    logger = {}
    np.random.seed(0)
    for epoch in tqdm(range(meta_training_steps)):  # Line 2 in the pseudocode

        tasks = sample_tasks(task_dist, n_tasks) # Line 3 in the pseudocode
        meta_optimizer.zero_grad()
        for task in tasks:
            if task.task_id not in logger:
                logger[task.task_id] = []
            task_training_log = fine_tune_meta(meta_model,
                                               task,
                                               inner_training_steps,
                                               alpha,
                                               meta_optimizer,
                                               device=device)
            logger[task.task_id].append(task_training_log)
        meta_optimizer.step()  # Line 10 in the pseudocode
    return logger