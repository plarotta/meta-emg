# cross-session redo with scaling
#python train.py 'test.run_name=scale_expt_minmax' 'test.stride=5' 'test.meta_steps=30' 'test.scale=2'
python train.py 'test.run_name=scale_expt_standardized' 'test.stride=5' 'test.meta_steps=30' 'test.scale=1'
python train.py 'test.run_name=scale_expt_noscale' 'test.stride=5' 'test.meta_steps=30' 'test.scale=0'
