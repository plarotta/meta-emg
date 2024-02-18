from source.models import BasicCNN, BasicDNN, TCN
import torch.optim as optim
from source.training_utils import maml, load_in_task_collection, get_save_dirs, get_baseline1, get_baseline2, process_logs, eval_trained_meta
from sklearn.model_selection import train_test_split
import numpy.random
from torch import manual_seed
import os
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from matplotlib import pyplot as plt

@hydra.main(version_base=None, config_path="data/conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    INNER_LR = cfg.test.inner_lr
    OUTER_LR = cfg.test.outer_lr
    META_STEPS = cfg.test.meta_steps
    INNER_STEPS = cfg.test.inner_steps
    N_VAL_TASKS = cfg.test.n_val_tasks
    N_TRAIN_TASKS = cfg.test.n_train_tasks
    TRAIN_PATH = cfg.test.train_collection_json
    TEST_PATH = cfg.test.test_collection_json
    OUT_ROOT = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir if cfg.test.save else None
    WANDB = False #cfg.test.wandb
    FC_UNITS = cfg.test.fc_units
    BATCH_SIZE = cfg.test.batch_size
    TIME_SEQ_LEN = cfg.test.time_seq_len
    STRIDE = cfg.test.stride
    RUN_NAME = cfg.test.run_name
    SCALE = cfg.test.scale
    DEVICE = cfg.test.device
    MODEL = cfg.test.model.lower()
    SEED = cfg.test.seed

    # SET SEEDS FOR REPRODUCIBILITY
    numpy.random.seed(SEED)
    manual_seed(SEED)

    wandb_logger = None

    # GET TRAIN-TEST-VAL SPLIT 
    task_colxn = load_in_task_collection(TRAIN_PATH,
                                         batch_size=BATCH_SIZE, 
                                         time_seq=TIME_SEQ_LEN, 
                                         stride=STRIDE,
                                         scale=SCALE)
    if N_VAL_TASKS > 0:
        train_colxn, val_clxn = train_test_split(task_colxn, test_size=N_VAL_TASKS)
    else:
        train_colxn = task_colxn
        val_clxn = task_colxn
    test_clxn = load_in_task_collection(TEST_PATH,
                                         batch_size=BATCH_SIZE, 
                                         time_seq=TIME_SEQ_LEN, 
                                         stride=STRIDE,
                                         scale=SCALE)
    
    # DEFINE MODELS 
    assert MODEL in ['cnn','dnn','tcn'], 'model must be one of [cnn,dnn,tcn]'
    if MODEL == 'dnn':
        meta_model = BasicDNN(seq_len=TIME_SEQ_LEN, dim1=128, dim2=279)
        b1_model = BasicDNN(seq_len=TIME_SEQ_LEN, dim1=128, dim2=279)
        b2_model = BasicDNN(seq_len=TIME_SEQ_LEN, dim1=128, dim2=279)
    elif MODEL == 'cnn':
        meta_model = BasicCNN(fc_dim=FC_UNITS, input_seq_len=TIME_SEQ_LEN)
        b1_model = BasicCNN(fc_dim=FC_UNITS, input_seq_len=TIME_SEQ_LEN)
        b2_model = BasicCNN(fc_dim=FC_UNITS, input_seq_len=TIME_SEQ_LEN)
    else:
        meta_model = TCN(8, 3*[26], TIME_SEQ_LEN, kernel_size=3)
        b1_model = TCN(8, 3*[26], TIME_SEQ_LEN, kernel_size=3)
        b2_model = TCN(8, 3*[26], TIME_SEQ_LEN, kernel_size=3)

    meta_model.load_state_dict(torch.load('/Users/plarotta/software/meta-emg/data/expt_outputs/2024-02-17/18-05-14/models/epoch_0014_loss_0.6338/model_state_dict.pth'))
    m_logs = eval_trained_meta(meta_model, test_clxn, INNER_STEPS, INNER_LR, device=DEVICE)

    # RUN BASELINES
    base1_logs = get_baseline1(b1_model, test_clxn, INNER_STEPS, INNER_LR, wandb_logger, device=DEVICE) # blank aka self
    base2_logs = get_baseline2(b2_model, train_colxn, test_clxn, INNER_STEPS, INNER_LR,device=DEVICE, wandb=wandb_logger, batch_size=BATCH_SIZE, stride=STRIDE, time_seq_len=TIME_SEQ_LEN, scale=SCALE) # pre training aka fine-tuned
    

    res_table, fig = process_logs(m_logs, base1_logs, base2_logs)
    res_table.to_csv(os.path.join('/Users/plarotta/software/meta-emg/','res_table.csv'))
    plt.show()


if __name__ == '__main__':
    main()
