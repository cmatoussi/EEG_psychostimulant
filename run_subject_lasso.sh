#!/bin/bash
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate
export PYTHONPATH="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/ml/new_ml/visualize:/home/mat/projects/coco-pipe:$PYTHONPATH"
cd /home/mat/projects/EEG_psychostimulant

echo "Running sensor subject..."
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_global_lasso.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_subject.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_subject.yml > lasso_subject_out.txt 2>&1

echo "Done"
