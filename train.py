from source.models import BasicCNN
import torch.optim as optim
from source.training_utils import maml, load_in_task_collection, get_save_dirs, get_baseline1, get_baseline2, process_logs
from sklearn.model_selection import train_test_split
import numpy.random
from torch import manual_seed
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import os
import wandb
import torch
import shutil


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

    wandb_logger = wandb.init(name='memg fc dim of 128') if WANDB else None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # ONLY SAVE CHECKPOINTS IF AN OUTPUT DIRECTORY NAME IS GIVEN
    if OUT_ROOT:
        MODEL_DIR, RES_DIR = get_save_dirs(OUT_ROOT)
    else:
        MODEL_DIR, RES_DIR = None, None

    # GET TEST-VAL SPLIT 
    task_colxn = load_in_task_collection(TC_PATH)
    train_colxn, val_clxn = train_test_split(task_colxn, test_size=N_VAL_TASKS)
    test_clxn = load_in_task_collection(r'C:\Users\plarotta\software\meta-emg\data\task_collections\patient_tc_test.json')


    # DEFINE MODEL + OPTIMIZER
    meta_model = BasicCNN(fc_dim=128)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=OUTER_LR)

    # RUN BASELINES ON TEST
    base1_logs = get_baseline1(BasicCNN(fc_dim=128), test_clxn, INNER_STEPS, INNER_LR, wandb_logger, device=device) # blank aka self
    base2_logs = get_baseline2(BasicCNN(fc_dim=128), train_colxn, test_clxn, INNER_STEPS, INNER_LR,device=device, wandb=wandb_logger) # pre training aka fine-tuned

    # # # # RUN MAML
    print("SETUP COMPLETE. BEGINNING MAML...")
    maml_logs = maml(meta_model, 
                     train_colxn,
                     val_clxn,
                     test_clxn,
                     meta_optimizer, 
                     INNER_STEPS, 
                     META_STEPS, 
                     INNER_LR, 
                     n_tasks=N_TRAIN_TASKS,
                     model_save_dir=MODEL_DIR,
                     wandb=wandb_logger)
    
    res_table, fig = process_logs(maml_logs, base1_logs, base2_logs)

    # ONLY SAVE CHECKPOINTS IF AN OUTPUT DIRECTORY NAME IS GIVEN
    if OUT_ROOT:
        with open(os.path.join(RES_DIR, 'maml_logger.pickle'), 'wb') as handle:
            pickle.dump(maml_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        res_table.to_csv(os.path.join(RES_DIR,'res_table.csv'))
        fig.savefig(os.path.join(RES_DIR,'res_barplot.png'))

    print(f"SUCCESSFULLY COMPLETED MAML RUN.")
    if wandb:
        print(RES_DIR)
        print(MODEL_DIR)
        print(wandb.run.dir)
        shutil.copytree(RES_DIR, os.path.join(wandb.run.dir,'res'))
        shutil.copytree(MODEL_DIR, os.path.join(wandb.run.dir,'models'))
        wandb_logger.finish()
        
    return(maml_logs)

if __name__ == '__main__':
    main()
