from source.models import BasicCNN
import torch.optim as optim
from source.training_utils import maml, load_in_task_collection, get_save_dirs, get_baseline1, get_baseline2
from sklearn.model_selection import train_test_split
import numpy.random
from torch import manual_seed
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import os
import wandb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch



@hydra.main(version_base=None, config_path="data/conf", config_name="config")
def main(cfg: DictConfig):

    # SET SEEDS FOR REPRODUCIBILITY
    numpy.random.seed(0)
    manual_seed(0)

    # PRINT OUT PARAMS & LOAD THEM
    print(OmegaConf.to_yaml(cfg))
    INNER_LR = cfg.test.inner_lr
    OUTER_LR = cfg.test.outer_lr
    META_STEPS = cfg.test.meta_steps
    INNER_STEPS = cfg.test.inner_steps
    N_VAL_TASKS = cfg.test.n_val_tasks
    N_TRAIN_TASKS = cfg.test.n_train_tasks
    TC_PATH = cfg.test.task_collection_json
    OUT_ROOT = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir if cfg.test.save else None
    WANDB = cfg.test.wandb

    wandb_logger = wandb.init(name='both baselines, 150 steps, true double adam, 1e-4 OLR') if WANDB else None

    # ONLY SAVE CHECKPOINTS IF THE OUT_ROOT IS GIVEN
    if OUT_ROOT:
        MODEL_DIR, RES_DIR = get_save_dirs(OUT_ROOT)
    else:
        MODEL_DIR, RES_DIR = None, None

    # GET TEST-VAL SPLIT 
    # task_colxn = load_in_task_collection(TC_PATH)

    # train_colxn, val_clxn = train_test_split(task_colxn, test_size=N_VAL_TASKS)
    train_colxn = load_in_task_collection(TC_PATH)
    val_clxn = load_in_task_collection(r'C:\Users\plarotta\software\meta-emg\data\task_collections\patient_tc_val.json')


    # DEFINE MODEL + OPTIMIZER
    meta_model = BasicCNN()
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=OUTER_LR)

    # # # # RUN MAML
    print("SETUP COMPLETE. BEGINNING MAML...")
    maml_logs = maml(meta_model, 
                     train_colxn,
                     val_clxn,
                     meta_optimizer, 
                     INNER_STEPS, 
                     META_STEPS, 
                     INNER_LR, 
                     n_tasks=N_TRAIN_TASKS,
                     model_save_dir=MODEL_DIR)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base1_logs = get_baseline1(BasicCNN(), val_clxn, INNER_STEPS, INNER_LR, wandb_logger, device=device) # blank
    base2_logs = get_baseline2(BasicCNN(), train_colxn, val_clxn, INNER_STEPS, INNER_LR,device=device) # pre training
    

    meta_accs = []
    meta_labs = []
    for t in maml_logs['val']:
        meta_accs.append(maml_logs['val'][t][-1]['val_accuracy'])
        meta_labs.append(t)

    
    b1_accs = []
    b1_labs = []
    for t in base1_logs['val']:
        b1_accs.append(base1_logs['val'][t][-1]['val_accuracy'])
        b1_labs.append(t)

    b2_accs = []
    b2_labs = []
    for t in base2_logs['val']:
        b2_accs.append(base2_logs['val'][t][-1]['val_accuracy'])
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
    plt.show()
    

    if OUT_ROOT:
        with open(os.path.join(RES_DIR, 'maml_logger.pickle'), 'wb') as handle:
            pickle.dump(maml_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        res_table.to_csv(os.path.join(RES_DIR,'res_table.csv'))
        fig.savefig(os.path.join(RES_DIR,'res_barplot.png'))

    print(f"SUCCESSFULLY COMPLETED MAML RUN.")
    return(maml_logs)

if __name__ == '__main__':
    main()
