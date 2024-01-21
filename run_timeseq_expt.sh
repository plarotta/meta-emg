# seq_len titration across 2 scaling conditions, then comparing models of different sizes

# std scaling
# python train.py 'test.run_name=time_seq_std_25' 'test.meta_steps=20' 'test.scale=1' 'test.time_seq_len=25'
python train.py 'test.run_name=time_seq_std_30' 'test.meta_steps=20' 'test.scale=1' 'test.time_seq_len=30' #model size 32
python train.py 'test.run_name=time_seq_std_35' 'test.meta_steps=20' 'test.scale=1' 'test.time_seq_len=35'
python train.py 'test.run_name=time_seq_std_40' 'test.meta_steps=20' 'test.scale=1' 'test.time_seq_len=40'
python train.py 'test.run_name=time_seq_std_45' 'test.meta_steps=20' 'test.scale=1' 'test.time_seq_len=45'
python train.py 'test.run_name=time_seq_std_50' 'test.meta_steps=20' 'test.scale=1' 'test.time_seq_len=50'

# minmax scaling
python train.py 'test.run_name=time_seq_minmax_25' 'test.meta_steps=20' 'test.scale=2' 'test.time_seq_len=25'
python train.py 'test.run_name=time_seq_minmax_30' 'test.meta_steps=20' 'test.scale=2' 'test.time_seq_len=30'
python train.py 'test.run_name=time_seq_minmax_35' 'test.meta_steps=20' 'test.scale=2' 'test.time_seq_len=35'
python train.py 'test.run_name=time_seq_minmax_40' 'test.meta_steps=20' 'test.scale=2' 'test.time_seq_len=40'
python train.py 'test.run_name=time_seq_minmax_45' 'test.meta_steps=20' 'test.scale=2' 'test.time_seq_len=45'
python train.py 'test.run_name=time_seq_minmax_50' 'test.meta_steps=20' 'test.scale=2' 'test.time_seq_len=50'

# model size expt
python train.py 'test.run_name=model_size_64_std_30' 'test.meta_steps=20' 'test.scale=1' 'test.time_seq_len=30' 'test.fc_units=64'