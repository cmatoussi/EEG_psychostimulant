import yaml
import os

config_path = "eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_subject.yml"
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
    
pooling = cfg.get("pooling")
representation = cfg.get("representation")

if pooling:
    spatial = "_region" if pooling == "region" else "_sensor"
else:
    spatial = ""

if representation:
    temporal = f"_{representation}"
else:
    temporal = ""

sp_type = "Region" if "region" in spatial else "Sensor"
tm_type = "Subject" if "subject" in temporal else "Epoch"

print(f"pooling: {pooling}")
print(f"representation: {representation}")
print(f"temporal: {temporal}")
print(f"spatial: {spatial}")
print(f"tm_type: {tm_type}")
print(f"sp_type: {sp_type}")
