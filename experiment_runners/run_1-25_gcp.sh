# fine tuning step experiment 
cd ../

#minmax
## 1 step
python train.py 'test.run_name=gpc_50_minmax_dnn_1step' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'

## 3 step
python train.py 'test.run_name=gpc_50_minmax_dnn_3step' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'

## 5 step
python train.py 'test.run_name=gpc_50_minmax_dnn_5step' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=5' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'

#std
## 1 step
python train.py 'test.run_name=gpc_50_minmax_dnn_1step' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'

## 3 step
python train.py 'test.run_name=gpc_50_minmax_dnn_3step' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'

## 5 step
python train.py 'test.run_name=gpc_50_minmax_dnn_5step' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=5' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'
