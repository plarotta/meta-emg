import re
import json

train_tasks = [
    '2023_02_17_p1/p1_111.csv', 
    '2023_02_17_p1/p1_121.csv',
    '2023_02_17_p1/p1_131.csv',
    '2023_02_17_p1/p1_141.csv',
    '2023_02_21_p3/p3_111.csv',
    '2023_02_21_p3/p3_121.csv',
    '2023_02_21_p3/p3_131.csv',
    '2023_02_21_p3/p3_141.csv',
    '2023_02_22_p4/p4_111.csv',
    '2023_02_22_p4/p4_121.csv',
    '2023_02_22_p4/p4_131.csv',
    '2023_02_22_p4/p4_141.csv',
    '2023_03_07_p7/p7_111.csv',
    '2023_03_07_p7/p7_121.csv',
    '2023_03_07_p7/p7_131.csv',
    '2023_03_07_p7/p7_141.csv',
    '2023_03_06_p8/p8_111.csv',
    '2023_03_06_p8/p8_121.csv',
    '2023_03_06_p8/p8_131.csv',
    '2023_03_06_p8/p8_141.csv',
    ]

test_tasks = [
    '2023_02_17_p1/p1_112.csv',
    '2023_02_17_p1/p1_122.csv',
    '2023_02_17_p1/p1_132.csv',
    '2023_02_17_p1/p1_142.csv',
    '2023_02_21_p3/p3_112.csv',
    '2023_02_21_p3/p3_122.csv',
    '2023_02_21_p3/p3_132.csv',
    '2023_02_21_p3/p3_142.csv',
    '2023_02_22_p4/p4_112.csv',
    '2023_02_22_p4/p4_122.csv',
    '2023_02_22_p4/p4_132.csv',
    '2023_02_22_p4/p4_142.csv',
    '2023_03_07_p7/p7_112.csv',
    '2023_03_07_p7/p7_122.csv',
    '2023_03_07_p7/p7_132.csv',
    '2023_03_07_p7/p7_142.csv',
    '2023_03_06_p8/p8_112.csv',
    '2023_03_06_p8/p8_122.csv',
    '2023_03_06_p8/p8_132.csv',
    '2023_03_06_p8/p8_142.csv'
]


train = []
for entry in train_tasks:
    train.append({'session': entry[:13],
               'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
print(train)

with open(r'C:\Users\plarotta\software\meta-emg\data\task_collections\stroke_set1_train.json', 'w') as json_file:
    json.dump(train, json_file)


test = []
for entry in test_tasks:
    test.append({'session': entry[:13],
               'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
print(test)

with open(r'C:\Users\plarotta\software\meta-emg\data\task_collections\stroke_set1_test.json', 'w') as json_file:
    json.dump(test, json_file)