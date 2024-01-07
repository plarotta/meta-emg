# M-EMG [WIP]

Meta-EMG implements Model-agnostic Meta-learning (MAML: https://arxiv.org/pdf/1703.03400.pdf) for the purpose of building models that can more quickly adapt to a patient.

Here we define a task as a single patient-condition instance represented by an 8-channel EMG recording, where condition is the 3-digit (111,112,131,etc) description of the procedure used to gather EMG data from them, and we define a task collection as a sampling of several tasks.

This repo contributes the following:

- A Python GUI for generating a task collection. Currently the GUI is pointed at the collected_data ROAM Lab repo, and it walks through all of the recordings in each session displaying the raw EMG signals as well as their corresponding ground-truth intent label. The GUI requests a save location at the end, but it is recommended to have it save to data/task_collections
- Dataset, task, and task collection data structures for convenient integration into meta learning frameworks
- A full MAML implementation based on the original TF implementation (https://github.com/cbfinn/maml) and the Higher module for higher-order gradients (https://github.com/facebookresearch/higher)
- Hydra integration so that experiments can be setup via YAML files and tracked easily

Before running anything, make sure you clone our environment from the YML file provided. 

To run the GUI to generate a new task collection,

```python source/task_space_generator.py```

## Running...

To run MAML, you need a task collection and a config file defining the hyperparameters and the location of the task collection. We provide a sample config file config.yml and its contents are displayed and defined below:

    test:
        inner_lr: 1e-4 (learning rate for the fine-tuning inner loop)
        outer_lr: 1e-4 (learning rate for the meta-update outer loop)
        meta_steps: 3 (number of meta updates to do)
        inner_steps: 15 (number of epochs for fine-tuning)
        n_val_tasks: 3 (number of hold-out tasks to use for meta-model validation)
        n_train_tasks: 3 (number of tasks to sample and fine-tune on for each meta update)
        task_collection_json: '/Users/plarotta/software/meta-emg/data/task_collections/pedro_ts1.json' (location of task collection)
        save: True (whether to save intermediate models and the logger dictionary)

    hydra:
        run:
            dir: data/expt_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S} (it's recommended you keep this so that the models and the results dict are saved in the same place Hydra automatically places the config file used for the run)


