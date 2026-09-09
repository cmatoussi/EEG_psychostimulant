#!/bin/bash

# Activate the environment
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate
export PYTHONPATH="/home/mat/projects/coco-pipe:$PYTHONPATH"
export MPLBACKEND=Agg
cd /home/mat/projects/EEG_psychostimulant/

echo "Generating Visualizations for Sensor Epoch..."
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_global_lasso.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_epoch.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_epoch.yml
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_per_feature.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_epoch.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_epoch.yml
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_per_sensor.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_epoch.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_epoch.yml

echo "Generating Visualizations for Sensor Subject..."
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_global_lasso.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_subject.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_subject.yml
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_per_feature.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_subject.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_subject.yml
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_per_sensor.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_subject.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_subject.yml

echo "Generating Visualizations for Pooled Epoch..."
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_global_lasso.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_pooled_epoch.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_epoch.yml
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_per_feature.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_pooled_epoch.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_epoch.yml
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_per_sensor.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_pooled_epoch.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_epoch.yml

echo "Generating Visualizations for Pooled Subject..."
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_global_lasso.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_pooled_subject.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_subject.yml
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_per_feature.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_pooled_subject.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_subject.yml
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_per_sensor.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_pooled_subject.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_subject.yml

echo "All Visualizations Generated!"
