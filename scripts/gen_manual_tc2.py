import json
import re

day2_patient_dict = [
    '2023_03_20_p1/p1_111_2.csv',
    '2023_03_20_p1/p1_111_longitudinal_non_integral_-0.5.csv',
    '2023_03_20_p1/p1_111_longitudinal_non_integral_0.5_2.csv',
    '2023_03_20_p1/p1_131.csv',
    '2023_03_20_p1/p1_131_longitudinal_non_integral_-0.5.csv',
    '2023_03_20_p1/p1_131_longitudinal_non_integral_0.5.csv',
    '2023_03_13_p3/p3_111.csv',
    '2023_03_13_p3/p3_111_longitudinal_non_integral_-0.5.csv',
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
    '2023_03_15_p8/p8_131_longitudinal_non_integral_0.5.csv'
    ]

tc = []
for entry in day2_patient_dict:
    tc.append({'session': entry[:13],
               'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
print(tc)

with open(r'C:\Users\plarotta\software\meta-emg\data\task_collections\patient_tc_val.json', 'w') as json_file:
    json.dump(tc, json_file)