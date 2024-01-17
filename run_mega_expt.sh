
# stride expt
python train.py 'test.run_name=stride_expt_stride_25_cross_sess' 'test.stride=25' 'test.meta_steps=10'
python train.py 'test.run_name=stride_expt_stride_20_cross_sess' 'test.stride=20' 'test.meta_steps=10'
python train.py 'test.run_name=stride_expt_stride_15_cross_sess' 'test.stride=15' 'test.meta_steps=10'
python train.py 'test.run_name=stride_expt_stride_10_cross_sess' 'test.stride=10' 'test.meta_steps=10'

# cross-patient @ stride=10
python train.py 'test.run_name=p4_long_run' 'test.train_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\p4_train.json' 'test.test_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\p4_test.json' 'test.meta_steps=15'

python train.py 'test.run_name=p3_long_run' 'test.train_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\p3_train.json' 'test.test_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\p3_test.json' 'test.meta_steps=15'

python train.py 'test.run_name=p7_long_run' 'test.train_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\p7_train.json' 'test.test_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\p7_test.json' 'test.meta_steps=15'

python train.py 'test.run_name=p8_long_run' 'test.train_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\p8_train.json' 'test.test_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\p8_test.json' 'test.meta_steps=15'

python train.py 'test.run_name=p1_long_run' 'test.train_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\p1_train.json' 'test.test_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\p1_test.json' 'test.meta_steps=15'


# cross-session redo with scaling
python train.py 'test.run_name=scale_expt' 'test.stride=10' 'test.meta_steps=15' 'test.scale=True'
python train.py 'test.run_name=scale_expt' 'test.stride=10' 'test.meta_steps=15' 'test.scale=False'
