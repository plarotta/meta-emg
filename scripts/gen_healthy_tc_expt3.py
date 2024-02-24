import re
import json
import os
import random


def main():
    pth = '/Users/plarotta/software/meta-emg/data/collected_data'
    dirs = os.listdir('/Users/plarotta/software/meta-emg/data/collected_data')
    dirs = [os.path.join(pth, i) for i in dirs if '.md' not in i and 'chat' not in i and 'gk2' not in i]

    dirs.sort(key=os.path.getmtime)
    dirs = set(dirs[33:]) # healthy subject sessions start here
    print(f'{len(dirs)} sessions')
    N_TRAINING_SESSIONS = [1,3,5,10]
    N_TEST_SESSIONS = 4
    N_TRIES=5

    # want to use the same test files for all runs so its a fair comparison
    test_paths = []
    test_dirs = set(random.sample(list(dirs), N_TEST_SESSIONS))
    for dir in test_dirs:
        for f in os.listdir(dir):
            if '.md' not in f:
                test_paths.append(os.path.join(dir,f)[54:])
    test = []
    for entry in test_paths:
        test.append({'session': entry[:13], 
                    'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
    ts_str = f'/Users/plarotta/software/meta-emg/data/task_collections/healthy/expt3/test.json'
    with open(ts_str, 'w') as json_file:
        json.dump(test, json_file)

    # do not include test dirs in the batch of dirs to sample from
    train_dirs = dirs.difference(test_dirs)

    for n in N_TRAINING_SESSIONS:
        for rep in range(N_TRIES):
            rep_dirs = random.sample(list(train_dirs), n)
            train_paths = []
            for dir in rep_dirs:
                for f in os.listdir(dir):
                    if '.md' not in f:
                        train_paths.append(os.path.join(dir,f)[54:])
            train = []
            for entry in train_paths:
                train.append({'session': entry[:13], 
                            'condition': re.search(r'_(.*?)\.csv', entry[13:]).group(1)})
            tr_str = f'/Users/plarotta/software/meta-emg/data/task_collections/healthy/expt3/{n}sessions_rep{rep}_train.json'
            with open(tr_str, 'w') as json_file:
                json.dump(train, json_file)

if __name__ == '__main__':
    main()
