# export WANDB_API_KEY=adf13475823af83082181ce3caf9b539bdf6bb5a

# conda create --name memg-env python=3.10 pip -y

# conda activate memg-env

# conda install -y conda install pytorch torchvision torchaudio cpuonly -c pytorch

# conda install -y scikit-learn scipy pandas matplotlib

# pip install wandb tqdm hydra-core higher

# git clone https://github.com/plarotta/meta-emg.git

# cd meta-emg/data && rm -rf collected_data

# git clone https://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@github.com/hand-orthosis/collected_data.git

# cd ../

# # 1 task batch size, time seq expt std vs minmax scaling vs model
# ##std
# ### cnn
# python train.py 'test.run_name=terremoto_timeseq_50_std_cnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=cnn'

# python train.py 'test.run_name=terremoto_timeseq_60_std_cnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=60' 'test.scale=1' 'test.model=cnn'

# python train.py 'test.run_name=terremoto_timeseq_70_std_cnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=70' 'test.scale=1' 'test.model=cnn'

# python train.py 'test.run_name=terremoto_timeseq_80_std_cnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=80' 'test.scale=1' 'test.model=cnn'

# python train.py 'test.run_name=terremoto_timeseq_90_std_cnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=90' 'test.scale=1' 'test.model=cnn'

# ### tcn
# python train.py 'test.run_name=terremoto_timeseq_50_std_tcn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=tcn'

# python train.py 'test.run_name=terremoto_timeseq_60_std_tcn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=60' 'test.scale=1' 'test.model=tcn'

# python train.py 'test.run_name=terremoto_timeseq_70_std_tcn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=70' 'test.scale=1' 'test.model=tcn'

# python train.py 'test.run_name=terremoto_timeseq_80_std_tcn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=80' 'test.scale=1' 'test.model=tcn'

# python train.py 'test.run_name=terremoto_timeseq_90_std_tcn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=90' 'test.scale=1' 'test.model=tcn'

# ### dnn
# python train.py 'test.run_name=terremoto_timeseq_50_std_dnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=1' 'test.model=dnn'

# python train.py 'test.run_name=terremoto_timeseq_60_std_dnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=60' 'test.scale=1' 'test.model=dnn'

# python train.py 'test.run_name=terremoto_timeseq_70_std_dnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=70' 'test.scale=1' 'test.model=dnn'

# python train.py 'test.run_name=terremoto_timeseq_80_std_dnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=80' 'test.scale=1' 'test.model=dnn'

# python train.py 'test.run_name=terremoto_timeseq_90_std_dnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=90' 'test.scale=1' 'test.model=dnn'


##minmax
### cnn
python train.py 'test.run_name=pedropc_timeseq_50_minmax_cnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=2' 'test.model=cnn'

python train.py 'test.run_name=pedropc_timeseq_60_minmax_cnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=60' 'test.scale=2' 'test.model=cnn'

python train.py 'test.run_name=pedropc_timeseq_70_minmax_cnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=70' 'test.scale=2' 'test.model=cnn'

python train.py 'test.run_name=pedropc_timeseq_80_minmax_cnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=80' 'test.scale=2' 'test.model=cnn'

python train.py 'test.run_name=pedropc_timeseq_90_minmax_cnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=90' 'test.scale=2' 'test.model=cnn'

### tcn
python train.py 'test.run_name=pedropc_timeseq_50_minmax_tcn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=2' 'test.model=tcn'

python train.py 'test.run_name=pedropc_timeseq_60_minmax_tcn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=60' 'test.scale=2' 'test.model=tcn'

python train.py 'test.run_name=pedropc_timeseq_70_minmax_tcn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=70' 'test.scale=2' 'test.model=tcn'

python train.py 'test.run_name=pedropc_timeseq_80_minmax_tcn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=80' 'test.scale=2' 'test.model=tcn'

python train.py 'test.run_name=pedropc_timeseq_90_minmax_tcn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=90' 'test.scale=2' 'test.model=tcn'

### dnn
python train.py 'test.run_name=pedropc_timeseq_50_minmax_dnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=50' 'test.scale=2' 'test.model=dnn'

python train.py 'test.run_name=pedropc_timeseq_60_minmax_dnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=60' 'test.scale=2' 'test.model=dnn'

python train.py 'test.run_name=pedropc_timeseq_70_minmax_dnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=70' 'test.scale=2' 'test.model=dnn'

python train.py 'test.run_name=pedropc_timeseq_80_minmax_dnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=80' 'test.scale=2' 'test.model=dnn'

python train.py 'test.run_name=pedropc_timeseq_90_minmax_dnn' 'test.inner_lr=1e-4' 'test.outer_lr=5e-4' 'test.meta_steps=30' 'test.inner_steps=3' 'test.n_val_tasks=5' 'test.n_train_tasks=1' 'test.save=True' 'test.wandb=True' 'test.time_seq_len=90' 'test.scale=2' 'test.model=dnn'


