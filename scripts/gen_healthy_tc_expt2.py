import os
import re
import json


def find_paths():
    '''helper to print out the files of interest for copy/paste convenience'''
    pth = '/Users/plarotta/software/meta-emg/data/collected_data'
        
    files = os.listdir('/Users/plarotta/software/meta-emg/data/collected_data')
    files = [os.path.join(pth, i) for i in files if '.md' not in i and 'chat' not in i and 'gk2' not in i]


    files.sort(key=os.path.getmtime)
    files = files[33:]
    print(f'{len(files)} files')

    train_paths = []
    test_paths = []
    train_strs = ['111.csv', '111_rotation_integral.csv','111_translation_non_integral.csv', '112.csv']
    test_strs = ['111_rotation_non_integral.csv']
    for dir in files:
        for f in os.listdir(dir):
            if f[3:] in train_strs:
                train_paths.append(os.path.join(dir,f))
            elif f[3:] in test_strs:
                test_paths.append(os.path.join(dir,f))
    print(f'num train files: {len(train_paths)}\n')
    print(f'num test files: {len(test_paths)}')
    print('train paths:\n')
    for f in train_paths:
        print(f'"{f}",')

    print('\ntest paths:\n')
    for f in test_paths:
        print(f'"{f}",')

def main():
    train_paths = [
        "2023_10_06_ac/ac_111.csv",
        "2023_10_06_ac/ac_112.csv",
        "2023_10_06_ac/ac_111_translation_non_integral.csv",
        "2023_10_06_ac/ac_111_rotation_integral.csv",
        "2023_10_06_ts/ts_111_translation_non_integral.csv",
        "2023_10_06_ts/ts_112.csv",
        "2023_10_06_ts/ts_111.csv",
        "2023_10_06_ts/ts_111_rotation_integral.csv",
        "2023_10_07_cl/cl_111_rotation_integral.csv",
        "2023_10_07_cl/cl_111.csv",
        "2023_10_07_cl/cl_112.csv",
        "2023_10_07_cl/cl_111_translation_non_integral.csv",
        "2023_10_07_dy/dy_111.csv",
        "2023_10_07_dy/dy_112.csv",
        "2023_10_07_dy/dy_111_translation_non_integral.csv",
        "2023_10_07_dy/dy_111_rotation_integral.csv",
        "2023_10_16_fa/fa_111.csv",
        "2023_10_16_fa/fa_112.csv",
        "2023_10_16_fa/fa_111_translation_non_integral.csv",
        "2023_10_16_fa/fa_111_rotation_integral.csv",
        "2023_10_19_wx/wx_111_translation_non_integral.csv",
        "2023_10_19_wx/wx_112.csv",
        "2023_10_19_wx/wx_111.csv",
        "2023_10_19_wx/wx_111_rotation_integral.csv",
        "2023_10_20_gk/gk_111.csv",
        "2023_10_20_gk/gk_112.csv",
        "2023_10_20_gk/gk_111_translation_non_integral.csv",
        "2023_10_20_gk/gk_111_rotation_integral.csv",
        "2023_10_20_jp/jp_111_translation_non_integral.csv",
        "2023_10_20_jp/jp_111_rotation_integral.csv",
        "2023_10_20_jp/jp_112.csv",
        "2023_10_20_jp/jp_111.csv",
        "2023_10_23_xw/xw_111_rotation_integral.csv",
        "2023_10_23_xw/xw_111.csv",
        "2023_10_23_xw/xw_112.csv",
        "2023_10_23_xw/xw_111_translation_non_integral.csv",
        "2023_10_23_yc/yc_112.csv",
        "2023_10_23_yc/yc_111.csv",
        "2023_10_23_yc/yc_111_rotation_integral.csv",
        "2023_10_23_yc/yc_111_translation_non_integral.csv",
        "2023_10_25_ae/ae_111_translation_non_integral.csv",
        "2023_10_25_ae/ae_111_rotation_integral.csv",
        "2023_10_25_ae/ae_111.csv",
        "2023_10_25_ae/ae_112.csv",
        "2023_10_27_jo/jo_111_translation_non_integral.csv",
        "2023_10_27_jo/jo_111_rotation_integral.csv",
        "2023_10_27_jo/jo_112.csv",
        "2023_10_27_jo/jo_111.csv",
        "2023_10_27_si/si_111_translation_non_integral.csv",
        "2023_10_27_si/si_111.csv",
        "2023_10_27_si/si_111_rotation_integral.csv",
        "2023_10_27_si/si_112.csv",
        "2023_11_02_as/as_111_rotation_integral.csv",
        "2023_11_02_as/as_111_translation_non_integral.csv",
        "2023_11_02_as/as_112.csv",
        "2023_11_02_as/as_111.csv",
        "2023_11_02_im/im_111.csv",
        "2023_11_02_im/im_112.csv",
        "2023_11_02_im/im_111_translation_non_integral.csv",
        "2023_11_02_im/im_111_rotation_integral.csv",
        "2023_11_03_hr/hr_111.csv",
        "2023_11_03_hr/hr_112.csv",
        "2023_11_03_hr/hr_111_rotation_integral.csv",
        "2023_11_03_hr/hr_111_translation_non_integral.csv",
        "2023_11_03_is/is_112.csv",
        "2023_11_03_is/is_111_translation_non_integral.csv",
        "2023_11_03_is/is_111.csv",
        "2023_11_03_is/is_111_rotation_integral.csv",
    ]

    test_paths = [
        "2023_10_06_ac/ac_111_rotation_non_integral.csv",
        "2023_10_06_ts/ts_111_rotation_non_integral.csv",
        "2023_10_07_cl/cl_111_rotation_non_integral.csv",
        "2023_10_07_dy/dy_111_rotation_non_integral.csv",
        "2023_10_16_fa/fa_111_rotation_non_integral.csv",
        "2023_10_19_wx/wx_111_rotation_non_integral.csv",
        "2023_10_20_gk/gk_111_rotation_non_integral.csv",
        "2023_10_20_jp/jp_111_rotation_non_integral.csv",
        "2023_10_23_xw/xw_111_rotation_non_integral.csv",
        "2023_10_23_yc/yc_111_rotation_non_integral.csv",
        "2023_10_25_ae/ae_111_rotation_non_integral.csv",
        "2023_10_27_jo/jo_111_rotation_non_integral.csv",
        "2023_10_27_si/si_111_rotation_non_integral.csv",
        "2023_11_02_as/as_111_rotation_non_integral.csv",
        "2023_11_02_im/im_111_rotation_non_integral.csv",
        "2023_11_03_hr/hr_111_rotation_non_integral.csv",
        "2023_11_03_is/is_111_rotation_non_integral.csv",

    ]

    train = []
    test = []
    for entry in train_paths:
        train.append({'session': entry[:13], 
                      'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
    for entry in test_paths:
        test.append({'session': entry[:13], 
                      'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
    
    tr_str = f'/Users/plarotta/software/meta-emg/data/task_collections/healthy/expt2_train.json'
    ts_str = f'/Users/plarotta/software/meta-emg/data/task_collections/healthy/expt2_test.json'
    with open(tr_str, 'w') as json_file:
        json.dump(train, json_file)

    with open(ts_str, 'w') as json_file:
        json.dump(test, json_file)

if __name__ == '__main__':
    main()