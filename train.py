from source.models import BasicCNN
import torch.optim as optim
from source.training_utils import maml, load_in_task_collection, get_save_dirs
from sklearn.model_selection import train_test_split
import numpy.random
from torch import manual_seed
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="/Users/plarotta/software/meta-emg/data/conf", config_name="conf")
def main(cfg: DictConfig):
    # SET SEEDS FOR REPRODUCIBILITY
    numpy.random.seed(0)
    manual_seed(0)

    # TODO: finish adding Hydra support to main()
    INNER_LR = cfg.inner_lr
    OUTER_LR = cfg.outer_lr
    META_STEPS = cfg.meta_steps
    INNER_STEPS = cfg.inner_steps
    N_VAL_TASKS = cfg.n_val_tasks
    N_TRAIN_TASKS = cfg.n_train_tasks
    TC_PATH = cfg.task_collection_json#'/Users/plarotta/software/meta-emg/data/task_collections/pedro_ts1.json'
    OUT_ROOT = cfg.out_root #'/Users/plarotta/software/meta-emg/data/expt_outputs'

    if OUT_ROOT:
        MODEL_DIR, RES_DIR, CONF_DIR = get_save_dirs(OUT_ROOT)

    
    task_colxn = load_in_task_collection(TC_PATH)
    train_colxn, val_clxn = train_test_split(task_colxn, test_size=N_VAL_TASKS)

    meta_model = BasicCNN()
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=OUTER_LR)

    maml_logs = maml(meta_model, 
                     train_colxn,
                     val_clxn,
                     meta_optimizer, 
                     INNER_STEPS, 
                     META_STEPS, 
                     INNER_LR, 
                     n_tasks=N_TRAIN_TASKS)
    
    return(maml_logs)

if __name__ == '__main__':
    main()
