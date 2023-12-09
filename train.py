from source.models import BasicCNN
import torch.optim as optim
from source.maml_utils import maml



def main():
    # PARAMS
    INNER_LR = 1e-4
    OUTER_LR = 1e-4
    META_STEPS = 3

    
    # TODO: create task distribution
    task_space = None

    meta_model = BasicCNN()
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=1e-3)

    expt_logs = maml(meta_model, task_space, meta_optimizer, INNER_LR, META_STEPS, OUTER_LR)

    return(expt_logs)

if __name__ == '__main__':
    main()
