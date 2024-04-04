# MetaEMG

## MyHand Orthosis

 Exotendon device and EMG armband           |  Fully sensorized orthosis
:-------------------------:|:-------------------------:
![image](https://github.com/plarotta/meta-emg/assets/20714356/f5ccd6c4-9db0-421d-aa9f-1665b4a1e7d5) | ![image](https://github.com/plarotta/meta-emg/assets/20714356/36014c4e-5c41-46ad-82d7-8971310af376) 
[ref](https://arxiv.org/pdf/1911.08003.pdf) | [ref](https://arxiv.org/pdf/2011.00034.pdf)

MyHand is a robotic orthosis designed to help with stroke rehabilitation. The actuators assist in opening and closing the user's hand via a series of linkages and actuators, and it is the intent of the user which ultimately drives the device. The intent is inferred from an EMG armband, and this process requires a model that takes in EMG as input and outputs an intent (open hand, close hand, relax hand) for the actuators. 

Training said model is difficult because there is significant patient-patient variation, and even within a patient there is a large amount of concept drift resulting from fatigue. This repo documents my exploration into meta-learning as a more effective training paradigm for this specific application.

## Meta learning

Meta-learning, or learning to learn, is a machine learning approach where models are trained to quickly adapt to new tasks with minimal data. It focuses on generalization across tasks by teaching models high-level skills or representations, enabling rapid learning and adaptation to new, unseen challenges. 

Model-agnostic meta learning (MAML) is an algorithm developed in [Finn et al.](https://arxiv.org/pdf/1703.03400.pdf) where a model is fine-tuned to individual tasks, and the aggregate loss of the fine-tuned models with respect to the initial model's parameters is what's used for the gradient step of the initial model. This initial model is the meta model. 

![image](https://github.com/plarotta/meta-emg/assets/20714356/abd212af-d1fa-4c38-ac59-c5725b00e537)

In principle, this training paradigm should result in a model that primarily learns task-agnostic features allowing it to fine-tune to a new task more easily (assuming the new task is similar enough to the training tasks). In our context, this means a training paradigm that can more quickly learn the user's EMG-intent mapping and is also somewhat robust to concept drift.

Furthermore, we define a task as a single patient-condition instance represented by an 8-channel EMG recording, where condition is the 3-digit (111,112,131,etc) description of the procedure used to gather EMG data from them, and we define a task collection as a sampling of several tasks.

This repo contributes the following:

- A Python GUI for generating a task collection. Currently the GUI is pointed at the collected_data ROAM Lab repo, and it walks through all of the recordings in each session displaying the raw EMG signals as well as their corresponding ground-truth intent label. The GUI requests a save location at the end, but it is recommended to have it save to data/task_collections
- Dataset, task, and task collection data structures for convenient integration into meta learning frameworks
- A full MAML implementation based on the original TF implementation (https://github.com/cbfinn/maml) and the Higher module for higher-order gradients (https://github.com/facebookresearch/higher)
- Hydra integration so that experiments can be setup via YAML files and tracked easily

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


