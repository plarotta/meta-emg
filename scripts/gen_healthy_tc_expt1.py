import json
import re

train_files  = [
    "2023_10_06_ac/ac_111.csv",
    "2023_10_06_ts/ts_111.csv",
    "2023_10_07_cl/cl_111.csv",
    "2023_10_07_dy/dy_111.csv",
    "2023_10_16_fa/fa_111.csv",
    "2023_10_19_wx/wx_111.csv",
    "2023_10_20_gk/gk_111.csv",
    "2023_10_20_jp/jp_111.csv",
    "2023_10_23_xw/xw_111.csv",
    "2023_10_23_yc/yc_111.csv",
    "2023_10_25_ae/ae_111.csv",
    "2023_10_27_jo/jo_111.csv",
    "2023_10_27_si/si_111.csv",
    "2023_11_02_as/as_111.csv",
    "2023_11_02_im/im_111.csv",
    "2023_11_03_hr/hr_111.csv",
    "2023_11_03_is/is_111.csv"
]


test_files = [
    "2023_10_06_ac/ac_112.csv",
    "2023_10_06_ts/ts_112.csv",
    "2023_10_07_cl/cl_112.csv",
    "2023_10_07_dy/dy_112.csv",
    "2023_10_16_fa/fa_112.csv",
    "2023_10_19_wx/wx_112.csv",
    "2023_10_20_gk/gk_112.csv",
    "2023_10_20_jp/jp_112.csv",
    "2023_10_23_xw/xw_112.csv",
    "2023_10_23_yc/yc_112.csv",
    "2023_10_25_ae/ae_112.csv",
    "2023_10_27_jo/jo_112.csv",
    "2023_10_27_si/si_112.csv",
    "2023_11_02_as/as_112.csv",
    "2023_11_02_im/im_112.csv",
    "2023_11_03_hr/hr_112.csv",
    "2023_11_03_is/is_112.csv"
]

def main(tr=train_files, ts=test_files):
    train = []
    test = []
    for entry in tr:
        train.append({'session': entry[:13], 
                      'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
    for entry in ts:
        test.append({'session': entry[:13], 
                      'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
    
    tr_str = f'/Users/plarotta/software/meta-emg/data/task_collections/healthy/expt1_train.json'
    ts_str = f'/Users/plarotta/software/meta-emg/data/task_collections/healthy/expt1_test.json'
    with open(tr_str, 'w') as json_file:
        json.dump(train, json_file)

    with open(ts_str, 'w') as json_file:
        json.dump(test, json_file)



if __name__ == '__main__':
    main()