########################
echo 'BEGINNING EXPERIMENT 1...'
# expt 1 code
python train.py 'test.run_name=gpc_healthy_expt1' 'test.train_collection_json=/home/plarottaone4/meta-emg/data/task_collections/healthy/expt1_train.json' 'test.test_collection_json=/home/plarottaone4/meta-emg/data/task_collections/healthy/expt1_test.json'
echo 'FINISHED EXPERIMENT 1'

#########################
echo 'BEGINNING EXPERIMENT 2...'
# expt2 code
python train.py 'test.run_name=gpc_healthy_expt2' 'test.train_collection_json=/home/plarottaone4/meta-emg/data/task_collections/healthy/expt2_train.json' 'test.test_collection_json=/home/plarottaone4/meta-emg/data/task_collections/healthy/expt2_test.json'
echo 'FINISHED EXPERIMENT 2'

#########################
echo 'BEGINNING EXPERIMENT 3...'
# Define the session numbers
sessions=(1 3 5 10)
# Iterate over each session
for session in "${sessions[@]}"; do
    echo "Processing ${session}sessions..."
    
    # Iterate over each repetition for the current session
    for rep in {0..4}; do
        filename="${session}sessions_rep${rep}_train.json"
        echo "Processing ${filename}..."
        python train.py "test.run_name=gpc_healthy_expt3_s${session}_r${rep}" "test.train_collection_json=/home/plarottaone4/meta-emg/data/task_collections/healthy/expt3/${filename}" 'test.test_collection_json=/home/plarottaone4/meta-emg/data/task_collections/healthy/expt3/test.json'

    done
done
echo 'FINISHED EXPERIMENT 3'

########################
echo 'BEGINNING EXPERIMENT 4...'
# expt4 code
python train.py 'test.run_name=gpc_healthy_expt4' 'test.train_collection_json=/home/plarottaone4/meta-emg/data/task_collections/healthy/expt4_train.json' 'test.test_collection_json=/home/plarottaone4/meta-emg/data/task_collections/healthy/expt4_test.json'
echo 'FINISHED EXPERIMENT 4'

#########################
echo 'RUN COMPLETE! :)'