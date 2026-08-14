#!/bin/bash

# Default values
LIMIT=20
JSON_DIR="/home/mat/projects/EEG_psychostimulant/data/results/eval/layer_eval"
CONDITION="EO_baseline"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --limit) LIMIT="$2"; shift ;;
        --condition) CONDITION="$2"; shift ;;
        -h|--help)
            echo "Usage: ./run_all_layers.sh [--limit NUM] [--condition COND]"
            echo "Example: ./run_all_layers.sh --limit 50 --condition EO_baseline"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=================================================="
echo "Evaluating CBraMod Layers (0 to 11)..."
echo "Subjects Limit: $LIMIT"
echo "Condition:      $CONDITION"
echo "Output JSONs:   $JSON_DIR"
echo "=================================================="

# Create the JSON directory if it doesn't exist
mkdir -p "$JSON_DIR"

# Activate environment
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# Loop over layers 3 to 12 of CBraMod (indices 2 to 11)
for LAYER_IDX in {2..11}; do
    JSON_FILE="$JSON_DIR/eval_layer_${LAYER_IDX}_${CONDITION}.json"
    
    if [ -f "$JSON_FILE" ]; then
        echo "--------------------------------------------------"
        echo ">>> Skipping Layer $LAYER_IDX (Already exists) <<<"
        echo "--------------------------------------------------"
        continue
    fi

    echo "--------------------------------------------------"
    echo ">>> Running Evaluation for Layer $LAYER_IDX <<<"
    echo "--------------------------------------------------"
    
    python eeg_adhd_epilepsy/dl/cbramod/cbramod_evaluate.py \
        --all_layers \
        --layer_idx $LAYER_IDX \
        --limit $LIMIT \
        --conditions "$CONDITION" \
        --representation epoch_flat \
        --json_dir "$JSON_DIR"
        
    echo "Finished Layer $LAYER_IDX. JSON saved to $JSON_FILE"
done

echo "=================================================="
echo "Aggregating layer results into a summary..."
echo "=================================================="

cat << 'EOF' > "$JSON_DIR/summarize_layers.py"
import os, json, sys, glob
import numpy as np

json_dir = sys.argv[1]
condition = sys.argv[2]
files = glob.glob(os.path.join(json_dir, f"eval_layer_*_{condition}.json"))

best_layer = -1
best_bal_acc = -1
best_obj = None

metrics = {'accuracy': [], 'balanced_accuracy': [], 'f1': []}

for f in files:
    with open(f, 'r') as handle:
        data = json.load(handle)
        
    acc = data.get("accuracy", 0)
    bal_acc = data.get("balanced accuracy", 0)
    f1 = data.get("f1", 0)
    
    metrics['accuracy'].append(acc)
    metrics['balanced_accuracy'].append(bal_acc)
    metrics['f1'].append(f1)
    
    # Track the best layer (based on balanced accuracy)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        layer_str = str(data.get("last_layer", ""))
        best_layer = layer_str.replace("layer_", "") if "layer_" in layer_str else layer_str
        best_obj = {"accuracy": acc, "balanced_accuracy": bal_acc, "f1": f1}

if not metrics['balanced_accuracy']:
    print("Warning: No evaluaton JSONs found to summarize.")
    sys.exit(0)

output = {
    "condition": condition,
    "best_layer_num": best_layer,
    "best_layer_accuracy": best_obj["accuracy"],
    "best_layer_balanced_accuracy": best_obj["balanced_accuracy"],
    "best_layer_f1": best_obj["f1"],
    "avg_accuracy": float(np.mean(metrics['accuracy'])),
    "avg_balanced_accuracy": float(np.mean(metrics['balanced_accuracy'])),
    "avg_f1": float(np.mean(metrics['f1'])),
    "std_accuracy": float(np.std(metrics['accuracy'])),
    "std_balanced_accuracy": float(np.std(metrics['balanced_accuracy'])),
    "std_f1": float(np.std(metrics['f1'])),
}

out_path = os.path.join(json_dir, "summary_layers.json")
with open(out_path, 'w') as f_out:
    json.dump(output, f_out, indent=4)
print(f"Summary written to {out_path}")
EOF

python "$JSON_DIR/summarize_layers.py" "$JSON_DIR" "$CONDITION"

echo "=================================================="
echo "All layers evaluated! Check your results in:"
echo "ls -l $JSON_DIR"
echo "=================================================="
