from source.models import BasicCNN
import torch.optim as optim
from source.training_utils import maml, load_in_task_collection, get_save_dirs
from sklearn.model_selection import train_test_split
import numpy.random
from torch import manual_seed
import hydra
from omegaconf import DictConfig, OmegaConf


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
    OUT_ROOT = cfg.test.out_root

    # ONLY SAVE CHECKPOINTS IF THE OUT_ROOT IS GIVEN
    if OUT_ROOT:
        MODEL_DIR, RES_DIR, CONF_DIR = get_save_dirs(OUT_ROOT)

    # GET TEST-VAL SPLIT 
    task_colxn = load_in_task_collection(TC_PATH)
    train_colxn, val_clxn = train_test_split(task_colxn, test_size=N_VAL_TASKS)

    # DEFINE MODEL + OPTIMIZER
    meta_model = BasicCNN()
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=OUTER_LR)

    # RUN MAML
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
