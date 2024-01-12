import json
import re

full_patient_file_list = [
    '2023_03_20_p1/p1_111_2.csv',
    '2023_03_20_p1/p1_111_longitudinal_non_integral_-0.5.csv',
    '2023_03_20_p1/p1_111_longitudinal_non_integral_0.5_2.csv',
    '2023_03_20_p1/p1_131.csv',
    '2023_03_20_p1/p1_131_longitudinal_non_integral_-0.5.csv',
    '2023_03_20_p1/p1_131_longitudinal_non_integral_0.5.csv',
    '2023_03_13_p3/p3_111.csv',
    '2023_03_13_p3/p3_111_longitudinal_non_integral_0.5.csv',
    '2023_03_13_p3/p3_131.csv',
    '2023_03_13_p3/p3_131_longitudinal_non_integral_-0.5.csv',
    '2023_03_13_p3/p3_131_longitudinal_non_integral_0.5.csv',
    '2023_03_14_p4/p4_111_2.csv',
    '2023_03_14_p4/p4_111_longitudinal_non_integral_-0.5.csv',
    '2023_03_14_p4/p4_111_longitudinal_non_integral_0.5.csv',
    '2023_03_14_p4/p4_131.csv',
    '2023_03_14_p4/p4_131_longitudinal_non_integral_-0.5.csv',
    '2023_03_14_p4/p4_131_longitudinal_non_integral_0.5.csv',
    '2023_03_28_p7/p7_111.csv',
    '2023_03_28_p7/p7_111_longitudinal_non_integral_-0.5.csv',
    '2023_03_28_p7/p7_111_longitudinal_non_integral_0.5.csv',
    '2023_03_28_p7/p7_131.csv',
    '2023_03_28_p7/p7_131_longitudinal_non_integral_-0.5.csv',
    '2023_03_28_p7/p7_131_longitudinal_non_integral_0.5.csv',
    '2023_03_15_p8/p8_111_2.csv',
    '2023_03_15_p8/p8_111_longitudinal_non_integral_-0.5.csv',
    '2023_03_15_p8/p8_111_longitudinal_non_integral_0.5.csv',
    '2023_03_15_p8/p8_131.csv',
    '2023_03_15_p8/p8_131_longitudinal_non_integral_-0.5.csv',
    '2023_03_15_p8/p8_131_longitudinal_non_integral_0.5.csv',
    '2023_02_17_p1/p1_111.csv', 
    '2023_02_17_p1/p1_121.csv',
    '2023_02_17_p1/p1_131.csv',
    '2023_02_17_p1/p1_141.csv',
    '2023_02_17_p1/p1_112.csv',
    '2023_02_17_p1/p1_122.csv',
    '2023_02_17_p1/p1_132.csv',
    '2023_02_17_p1/p1_142.csv',
    '2023_02_21_p3/p3_111.csv',
    '2023_02_21_p3/p3_121.csv',
    '2023_02_21_p3/p3_131.csv',
    '2023_02_21_p3/p3_141.csv',
    '2023_02_21_p3/p3_112.csv',
    '2023_02_21_p3/p3_122.csv',
    '2023_02_21_p3/p3_132.csv',
    '2023_02_21_p3/p3_142.csv',
    '2023_02_22_p4/p4_111.csv',
    '2023_02_22_p4/p4_121.csv',
    '2023_02_22_p4/p4_131.csv',
    '2023_02_22_p4/p4_141.csv',
    '2023_02_22_p4/p4_112.csv',
    '2023_02_22_p4/p4_122.csv',
    '2023_02_22_p4/p4_132.csv',
    '2023_02_22_p4/p4_142.csv',
    '2023_03_07_p7/p7_111.csv',
    '2023_03_07_p7/p7_121.csv',
    '2023_03_07_p7/p7_131.csv',
    '2023_03_07_p7/p7_141.csv',
    '2023_03_07_p7/p7_112.csv',
    '2023_03_07_p7/p7_122.csv',
    '2023_03_07_p7/p7_132.csv',
    '2023_03_07_p7/p7_142.csv',
    '2023_03_06_p8/p8_111.csv',
    '2023_03_06_p8/p8_121.csv',
    '2023_03_06_p8/p8_131.csv',
    '2023_03_06_p8/p8_141.csv',
    '2023_03_06_p8/p8_112.csv',
    '2023_03_06_p8/p8_122.csv',
    '2023_03_06_p8/p8_132.csv',
    '2023_03_06_p8/p8_142.csv'
    ]

for p in ['p1', 'p3', 'p4', 'p7', 'p8']:

    train = []
    test = []
    for entry in full_patient_file_list:
        if p not in entry:
            print('train', p, entry)
            train.append({'session': entry[:13],
                    'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
        else:
            print('test', p, entry)
            test.append({'session': entry[:13],
                    'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})

    tr_str = f'C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\{p}_train.json'
    ts_str = f'C:\\Users\\plarotta\\software\\meta-emg\\data\\task_collections\\cross_patient\\{p}_test.json'
    with open(tr_str, 'w') as json_file:
        json.dump(train, json_file)
    
    with open(ts_str, 'w') as json_file:
        json.dump(test, json_file)