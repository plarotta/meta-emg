from source.models import BasicCNN, TCN, BasicDNN, TransformerNet
import torch.optim as optim
from source.training_utils import *
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
import random

@hydra.main(version_base=None, config_path="data/conf", config_name="config")
def main(cfg: DictConfig):

    # PRINT OUT PARAMS & LOAD THEM
    SEED = random.randint(0,10000) if cfg.test.seed == 'r' else cfg.test.seed
    print(f'ACTUAL SEED: {SEED}')
    INNER_LR = cfg.test.inner_lr
    OUTER_LR = cfg.test.outer_lr
    META_STEPS = cfg.test.meta_steps
    INNER_STEPS = cfg.test.inner_steps
    N_VAL_TASKS = cfg.test.n_val_tasks
    N_TRAIN_TASKS = cfg.test.n_train_tasks
    TRAIN_PATH = cfg.test.train_collection_json
    TEST_PATH = cfg.test.test_collection_json
    OUT_ROOT = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir if cfg.test.save else None
    WANDB = cfg.test.wandb
    FC_UNITS = cfg.test.fc_units
    BATCH_SIZE = cfg.test.batch_size
    TIME_SEQ_LEN = cfg.test.time_seq_len
    STRIDE = cfg.test.stride
    RUN_NAME = cfg.test.run_name
    SCALE = cfg.test.scale # 2 for minmax, 1 for std
    DEVICE = cfg.test.device
    MODEL = cfg.test.model.lower()
    RORCR_SIZE = cfg.test.rorcr_size # 1 for full rorcr
    LR_CYCLE = cfg.test.lr_cycle
    LR_REDUCE_FACTOR = cfg.test.lr_factor
    RUN_CONVERGENCE_TEST = cfg.test.converge_baselines

    # SPIN UP WANDB RUN
    wandb_logger = wandb.init(name=f'{RUN_NAME}') if WANDB else None
    
    # SET SEEDS FOR REPRODUCIBILITY
    numpy.random.seed(SEED)
    manual_seed(SEED)

    # DEVICE HANDLING
    if DEVICE is None:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif DEVICE == 'cuda' or DEVICE == 'cuda:0':
        assert torch.cuda.is_available(), 'gpu requested but not available... switch device to cpu'
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    print(DEVICE)
    print(OmegaConf.to_yaml(cfg))

    # ONLY SAVE CHECKPOINTS IF AN OUTPUT DIRECTORY NAME IS GIVEN
    if OUT_ROOT is not None:
        MODEL_DIR, RES_DIR = get_save_dirs(OUT_ROOT)
    else:
        MODEL_DIR, RES_DIR = None, None

    # GET TRAIN-TEST-VAL SPLIT 
    task_colxn = load_in_task_collection(TRAIN_PATH,
                                         batch_size=BATCH_SIZE, 
                                         time_seq=TIME_SEQ_LEN, 
                                         stride=STRIDE,
                                         scale=SCALE,
                                         rorcr_sample_size=RORCR_SIZE)
    test_clxn = load_in_task_collection(TEST_PATH,
                                        batch_size=BATCH_SIZE, 
                                        time_seq=TIME_SEQ_LEN, 
                                        stride=STRIDE,
                                        scale=SCALE,
                                        rorcr_sample_size=RORCR_SIZE)
    if N_VAL_TASKS > 0:
        train_colxn, val_clxn = train_test_split(task_colxn, test_size=N_VAL_TASKS)
    else:
        train_colxn = task_colxn
        val_clxn = test_clxn
    
    print("DATA LOAD-IN SUCCESSFUL\n")

    # DEFINE MODELS 
    assert MODEL in ['cnn','dnn','tcn','tf'], 'model must be one of [cnn,dnn,tcn,tf]'
    if MODEL == 'dnn':
        meta_model = BasicDNN(seq_len=TIME_SEQ_LEN, dim1=512, dim2=128)
        b1_model = BasicDNN(seq_len=TIME_SEQ_LEN, dim1=512, dim2=128)
        b2_model = BasicDNN(seq_len=TIME_SEQ_LEN, dim1=512, dim2=128)
    elif MODEL == 'cnn':
        meta_model = BasicCNN(fc_dim=FC_UNITS, input_seq_len=TIME_SEQ_LEN)
        b1_model = BasicCNN(fc_dim=FC_UNITS, input_seq_len=TIME_SEQ_LEN)
        b2_model = BasicCNN(fc_dim=FC_UNITS, input_seq_len=TIME_SEQ_LEN)
    elif MODEL == 'tcn':
        meta_model = TCN(8, 2*[TIME_SEQ_LEN+1], TIME_SEQ_LEN, kernel_size=3)
        b1_model = TCN(8, 2*[TIME_SEQ_LEN+1], TIME_SEQ_LEN, kernel_size=3)
        b2_model = TCN(8, 2*[TIME_SEQ_LEN+1], TIME_SEQ_LEN, kernel_size=3)
    else:
        meta_model = TransformerNet(seq_len=TIME_SEQ_LEN)
        b1_model = TransformerNet(seq_len=TIME_SEQ_LEN)
        b2_model = TransformerNet(seq_len=TIME_SEQ_LEN)

    # SPIN UP META OPTIMIZER
    meta_optimizer = optim.AdamW(meta_model.parameters(), lr=OUTER_LR)

    # RUN MAML
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
                     wandb=wandb_logger,
                     device=DEVICE,
                     lr_sched_cycle=LR_CYCLE,
                     lr_sched_gamma=LR_REDUCE_FACTOR)

    if MODEL_DIR is not None:
        shutil.copytree(MODEL_DIR, os.path.join(wandb.run.dir,'models'))

    # RUN BASELINES
    base1_logs = get_baseline1(b1_model, test_clxn, INNER_STEPS, INNER_LR, wandb_logger, device=DEVICE, save_model=MODEL_DIR) # blank aka self
    base2_logs = get_baseline2(b2_model, train_colxn, test_clxn, INNER_STEPS, INNER_LR,device=DEVICE, wandb=wandb_logger, batch_size=BATCH_SIZE, stride=STRIDE, time_seq_len=TIME_SEQ_LEN, scale=SCALE, save_model=MODEL_DIR) # pre training aka fine-tuned

    # # # # GENERATE TEST RESULTS
    res_table, fig = process_logs(maml_logs, base1_logs, base2_logs)

    # # # ONLY SAVE CHECKPOINTS IF AN OUTPUT DIRECTORY NAME IS GIVEN
    if OUT_ROOT is not None:
        with open(os.path.join(RES_DIR, 'maml_logger.pickle'), 'wb') as handle:
            pickle.dump(maml_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        res_table.to_csv(os.path.join(RES_DIR,'full_result_table.csv'))
        fig.savefig(os.path.join(RES_DIR,'res_barplot.png'))
    
    if RUN_CONVERGENCE_TEST is True:
        print('B1 CONVERGENCE')
        res = model_convergence_test(b1_model, 
                                     os.path.join(MODEL_DIR, 'b1_model_state_dict.pth'), 
                                     test_clxn, 
                                     lr=INNER_LR, 
                                     save_dir=RES_DIR,
                                     name='b1')

        print('B2 CONVERGENCE')
        res = model_convergence_test(b2_model, 
                                     os.path.join(MODEL_DIR, 'b2_model_state_dict.pth'), 
                                     test_clxn, 
                                     lr=INNER_LR, 
                                     save_dir=RES_DIR,
                                     name='b2')
    
    print(f"SUCCESSFULLY COMPLETED FULL EXPERIMENT.")
    if WANDB is True:
        shutil.copytree(RES_DIR, os.path.join(wandb.run.dir,'res'))
        wandb_logger.finish()
        
    return(maml_logs)

if __name__ == '__main__':
    main()
