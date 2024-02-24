import re
import json

# n training patients expt: the goal is to see if learning a new patient becomes
# easier the more patients that you meta

def save_tc(files, num_str, save_name_str):
    tc = []
    for entry in files:
        tc.append({'session': entry[:13], 
                    'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
    pth = f'/Users/plarotta/software/meta-emg/data/task_collections/num_patients_expt/{num_str}/{save_name_str}.json'
    with open(pth, 'w') as json_file:
        json.dump(tc, json_file)

    print("SAVED SPLITS")

file_dict = {
    'p1': [
        '2023_03_20_p1/p1_111_2.csv',
        '2023_03_20_p1/p1_111_longitudinal_non_integral_-0.5.csv',
        '2023_03_20_p1/p1_111_longitudinal_non_integral_0.5_2.csv',
        '2023_03_20_p1/p1_131.csv',
        '2023_03_20_p1/p1_131_longitudinal_non_integral_-0.5.csv',
        '2023_03_20_p1/p1_131_longitudinal_non_integral_0.5.csv',
        '2023_02_17_p1/p1_111.csv', 
        '2023_02_17_p1/p1_121.csv',
        '2023_02_17_p1/p1_131.csv',
        '2023_02_17_p1/p1_141.csv',
        '2023_02_17_p1/p1_112.csv',
        '2023_02_17_p1/p1_122.csv',
        '2023_02_17_p1/p1_132.csv',
        '2023_02_17_p1/p1_142.csv'],
    'p3':[
        '2023_03_13_p3/p3_111.csv',
        '2023_03_13_p3/p3_111_longitudinal_non_integral_0.5.csv',
        '2023_03_13_p3/p3_131.csv',
        '2023_03_13_p3/p3_131_longitudinal_non_integral_-0.5.csv',
        '2023_03_13_p3/p3_131_longitudinal_non_integral_0.5.csv',
        '2023_02_21_p3/p3_111.csv',
        '2023_02_21_p3/p3_121.csv',
        '2023_02_21_p3/p3_131.csv',
        '2023_02_21_p3/p3_141.csv',
        '2023_02_21_p3/p3_112.csv',
        '2023_02_21_p3/p3_122.csv',
        '2023_02_21_p3/p3_132.csv',
        '2023_02_21_p3/p3_142.csv'],
    'p4': [
        '2023_03_14_p4/p4_111_2.csv',
        '2023_03_14_p4/p4_111_longitudinal_non_integral_-0.5.csv',
        '2023_03_14_p4/p4_111_longitudinal_non_integral_0.5.csv',
        '2023_03_14_p4/p4_131.csv',
        '2023_03_14_p4/p4_131_longitudinal_non_integral_-0.5.csv',
        '2023_03_14_p4/p4_131_longitudinal_non_integral_0.5.csv',
        '2023_02_22_p4/p4_111.csv',
        '2023_02_22_p4/p4_121.csv',
        '2023_02_22_p4/p4_131.csv',
        '2023_02_22_p4/p4_141.csv',
        '2023_02_22_p4/p4_112.csv',
        '2023_02_22_p4/p4_122.csv',
        '2023_02_22_p4/p4_132.csv',
        '2023_02_22_p4/p4_142.csv'],
    'p7': [
        '2023_03_28_p7/p7_111.csv',
        '2023_03_28_p7/p7_111_longitudinal_non_integral_-0.5.csv',
        '2023_03_28_p7/p7_111_longitudinal_non_integral_0.5.csv',
        '2023_03_28_p7/p7_131.csv',
        '2023_03_28_p7/p7_131_longitudinal_non_integral_-0.5.csv',
        '2023_03_28_p7/p7_131_longitudinal_non_integral_0.5.csv',
        '2023_03_07_p7/p7_111.csv',
        '2023_03_07_p7/p7_121.csv',
        '2023_03_07_p7/p7_131.csv',
        '2023_03_07_p7/p7_141.csv',
        '2023_03_07_p7/p7_112.csv',
        '2023_03_07_p7/p7_122.csv',
        '2023_03_07_p7/p7_132.csv',
        '2023_03_07_p7/p7_142.csv'],
    'p8': [
        '2023_03_15_p8/p8_111_2.csv',
        '2023_03_15_p8/p8_111_longitudinal_non_integral_-0.5.csv',
        '2023_03_15_p8/p8_111_longitudinal_non_integral_0.5.csv',
        '2023_03_15_p8/p8_131.csv',
        '2023_03_15_p8/p8_131_longitudinal_non_integral_-0.5.csv',
        '2023_03_15_p8/p8_131_longitudinal_non_integral_0.5.csv',
        '2023_03_06_p8/p8_111.csv',
        '2023_03_06_p8/p8_121.csv',
        '2023_03_06_p8/p8_131.csv',
        '2023_03_06_p8/p8_141.csv',
        '2023_03_06_p8/p8_112.csv',
        '2023_03_06_p8/p8_122.csv',
        '2023_03_06_p8/p8_132.csv',
        '2023_03_06_p8/p8_142.csv']
}

p_list = ['p1','p3','p4','p7','p8']

for test_p in p_list:
    train_candidates = [i for i in p_list if i != test_p]
    test_f = file_dict[test_p]
    save_tc(test_f, 'test', f'{test_p}_test')

    # 1 training patient
    for i in range(len(train_candidates)):
        train_p = train_candidates[i]
        train_f = []
        train_f = file_dict[train_p]
        save_tc(train_f, 'one', f'{train_p}_train')
        # print(f'training on {train_p} testing on {test_p}')


    # 2 training patient
    for i in range(len(train_candidates)):
        for j in range(i+1, len(train_candidates)):
            train_p = [train_candidates[i],train_candidates[j]]
            train_f = []
            for p in train_p:
                train_f = train_f + file_dict[p]
            save_tc(train_f, 'two', f'{train_p[0]}-{train_p[1]}_train')
            # print(f'training on {train_p[0]}-{train_p[1]} testing on {test_p}')

    # 3 training patient
    # lazy so I'll just do these manually
    train_p = [train_candidates[0],train_candidates[1],train_candidates[2]]
    train_f = []
    for p in train_p:
        train_f = train_f + file_dict[p]
    save_tc(train_f, 'three', f'{train_p[0]}-{train_p[1]}-{train_p[2]}_train') 
    # print(f'training on {train_p[0]}-{train_p[1]}-{train_p[2]} testing on {test_p}')
          

    train_p = [train_candidates[1],train_candidates[2],train_candidates[3]]
    train_f = []
    for p in train_p:
        train_f = train_f + file_dict[p]  
    save_tc(train_f, 'three', f'{train_p[0]}-{train_p[1]}-{train_p[2]}_train') 
    # print(f'training on {train_p[0]}-{train_p[1]}-{train_p[2]} testing on {test_p}')   

    train_p = [train_candidates[0],train_candidates[2],train_candidates[3]]
    train_f = []
    for p in train_p:
        train_f = train_f + file_dict[p]  
    save_tc(train_f, 'three', f'{train_p[0]}-{train_p[1]}-{train_p[2]}_train')  
    # print(f'training on {train_p[0]}-{train_p[1]}-{train_p[2]} testing on {test_p}')    
                
    train_p = [train_candidates[0],train_candidates[1],train_candidates[3]]
    train_f = []
    for p in train_p:
        train_f = train_f + file_dict[p] 
    save_tc(train_f, 'three', f'{train_p[0]}-{train_p[1]}-{train_p[2]}_train')  
    # print(f'training on {train_p[0]}-{train_p[1]}-{train_p[2]} testing on {test_p}') 
    
    # 4 training patient
    train_p = train_candidates
    train_f = []
    for p in train_p:
        train_f = train_f + file_dict[p] 
    save_tc(train_f, 'four', f'{train_p[0]}-{train_p[1]}-{train_p[2]}-{train_p[3]}_train')  
    # print(f'training on {train_p[0]}-{train_p[1]}-{train_p[2]}-{train_p[3]} testing on {test_p}') 
    


        
