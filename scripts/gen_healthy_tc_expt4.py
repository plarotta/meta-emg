import re
import json
import os

def main():
    
    pth = '/Users/plarotta/software/meta-emg/data/collected_data'
    dirs = os.listdir('/Users/plarotta/software/meta-emg/data/collected_data')
    dirs = [os.path.join(pth, i) for i in dirs if '.md' not in i and 'chat' not in i and 'gk2' not in i]

    dirs.sort(key=os.path.getmtime)
    dirs = set(dirs[33:]) # healthy subject sessions start here
    print(len(dirs))

    train = []
    for dir in dirs:
        for f in os.listdir(dir):
            if '.md' not in f:
                entry = os.path.join(dir,f)[54:]
                train.append({'session': entry[:13], 'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
    tr_str = f'/Users/plarotta/software/meta-emg/data/task_collections/healthy/expt4_train.json'
    with open(tr_str, 'w') as json_file:
        json.dump(train, json_file)


    test_paths = [
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

    test = []
    for entry in test_paths:
        test.append({'session': entry[:13], 
                    'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
    ts_str = f'/Users/plarotta/software/meta-emg/data/task_collections/healthy/expt4_test.json'
    with open(ts_str, 'w') as json_file:
        json.dump(test, json_file)


if __name__ == '__main__':
    main()