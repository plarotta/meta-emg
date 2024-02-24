python train.py 'test.model=cnn' 'test.run_name=cross-sess_cnn' 'test.train_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\patient_tc_train.json' 'test.test_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\patient_tc_test.json'


python train.py 'test.model=tcn' 'test.run_name=cross-sess_tcn' 'test.train_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\patient_tc_train.json' 'test.test_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\patient_tc_test.json'


python train.py 'test.device=cpu' 'test.model=tf' 'test.run_name=cross-sess_tf' 'test.train_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\patient_tc_train.json' 'test.test_collection_json=C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\patient_tc_test.json'
