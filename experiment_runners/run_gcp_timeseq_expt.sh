# window size titration, 3 models, std scaling, 1 and 3 inner steps

cd ../
##cnn expts
### 50 
#### 3 inner steps
python train.py 'test.run_name=gcp_timeseq_50_cnn_3step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=cnn'
python train.py 'test.run_name=gcp_timeseq_50_cnn_3step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=cnn'

#### 1 inner steps
python train.py 'test.run_name=gcp_timeseq_50_cnn_1step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=cnn'
python train.py 'test.run_name=gcp_timeseq_50_cnn_1step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=cnn'

### 100 
#### 3 inner steps
python train.py 'test.run_name=gcp_timeseq_100_cnn_3step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=cnn'
python train.py 'test.run_name=gcp_timeseq_100_cnn_3step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=cnn'

#### 1 inner steps
python train.py 'test.run_name=gcp_timeseq_100_cnn_1step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=cnn'
python train.py 'test.run_name=gcp_timeseq_100_cnn_1step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=cnn'

### 150 
#### 3 inner steps
python train.py 'test.run_name=gcp_timeseq_150_cnn_3step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=cnn'
python train.py 'test.run_name=gcp_timeseq_150_cnn_3step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=cnn'

#### 1 inner steps
python train.py 'test.run_name=gcp_timeseq_150_cnn_1step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=cnn'
python train.py 'test.run_name=gcp_timeseq_150_cnn_1step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=cnn'

##tcn expts
### 50 
#### 3 inner steps
python train.py 'test.run_name=gcp_timeseq_50_tcn_3step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=tcn'
python train.py 'test.run_name=gcp_timeseq_50_tcn_3step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=tcn'

#### 1 inner steps
python train.py 'test.run_name=gcp_timeseq_50_tcn_1step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=cnn'
python train.py 'test.run_name=gcp_timeseq_50_tcn_1step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=cnn'

### 100 
#### 3 inner steps
python train.py 'test.run_name=gcp_timeseq_100_tcn_3step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=tcn'
python train.py 'test.run_name=gcp_timeseq_100_tcn_3step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=tcn'

#### 1 inner steps
python train.py 'test.run_name=gcp_timeseq_100_tcn_1step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=tcn'
python train.py 'test.run_name=gcp_timeseq_100_tcn_1step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=tcn'

### 150 
#### 3 inner steps
python train.py 'test.run_name=gcp_timeseq_150_tcn_3step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=tcn'
python train.py 'test.run_name=gcp_timeseq_150_tcn_3step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=tcn'

#### 1 inner steps
python train.py 'test.run_name=gcp_timeseq_150_tcn_1step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=tcn'
python train.py 'test.run_name=gcp_timeseq_150_tcn_1step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=tcn'


##dnn expts
### 50 
#### 3 inner steps
python train.py 'test.run_name=gcp_timeseq_50_dnn_3step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'
python train.py 'test.run_name=gcp_timeseq_50_dnn_3step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'

#### 1 inner steps
python train.py 'test.run_name=gcp_timeseq_50_dnn_1step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'
python train.py 'test.run_name=gcp_timeseq_50_dnn_1step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'

### 100 
#### 3 inner steps
python train.py 'test.run_name=gcp_timeseq_100_dnn_3step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=dnn'
python train.py 'test.run_name=gcp_timeseq_100_dnn_3step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=dnn'

#### 1 inner steps
python train.py 'test.run_name=gcp_timeseq_100_dnn_1step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=dnn'
python train.py 'test.run_name=gcp_timeseq_100_dnn_1step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=100' 'test.scale=1' 'test.model=dnn'

### 150 
#### 3 inner steps
python train.py 'test.run_name=gcp_timeseq_150_dnn_3step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=dnn'
python train.py 'test.run_name=gcp_timeseq_150_dnn_3step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=3' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=dnn'

#### 1 inner steps
python train.py 'test.run_name=gcp_timeseq_150_dnn_1step_rep1' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=dnn'
python train.py 'test.run_name=gcp_timeseq_150_dnn_1step_rep2' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=20' 'test.inner_steps=1' 'test.n_val_tasks=3' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=150' 'test.scale=1' 'test.model=dnn'
