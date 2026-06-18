#!/usr/bin/env python3
import json
import yaml
import csv
from pathlib import Path

def main():
    config_path = Path(__file__).parent / "experiments_config.yaml"
    if not config_path.exists():
        config_path = Path("experiments_config.yaml")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    groups = config.get("groups", {})
    
    # Extract options excluding 'ALL'
    sex_opts = [x for x in groups.get("sex", []) if x != "ALL"]
    age_opts = [x for x in groups.get("age", []) if x != "ALL"]
    com_opts = [x for x in groups.get("comorbidities", []) if x != "ALL"]
    med_opts = [x for x in groups.get("medication", []) if x != "ALL"]
    
    # Generate the 15 distinct cohorts (filter by one variable at a time)
    cohorts = []
    
    # 1. Default (ALL, ALL, ALL, ALL)
    cohorts.append(("ALL", "ALL", "ALL", "ALL"))
    
    # 2. Filter by Sex (others are ALL)
    for sex in sex_opts:
        cohorts.append((sex, "ALL", "ALL", "ALL"))
        
    # 3. Filter by Age (others are ALL)
    for age in age_opts:
        cohorts.append(("ALL", age, "ALL", "ALL"))
        
    # 4. Filter by Comorbidities (others are ALL)
    for com in com_opts:
        cohorts.append(("ALL", "ALL", com, "ALL"))
        
    # 5. Filter by Medication (others are ALL)
    for med in med_opts:
        cohorts.append(("ALL", "ALL", "ALL", med))
        
    # 28 distinct analyses
    analyses = []
    
    # 1. Dimension reduction (3 methods)
    for method in ["PCA", "UMAP", "t-SNE"]:
        analyses.append({
            "type": "dim_reduction",
            "method": method,
            "level": "subject",
            "balance": "balanced"
        })
        
    # 2. Handcrafted (3 units x 3 regression heads = 9)
    for unit in ["feature", "sensor", "region"]:
        for reg_head in ["SVC", "LogisticRegression", "RandomForestClassifier"]:
            analyses.append({
                "type": "handcrafted",
                "unit": unit,
                "regression_head": reg_head,
                "level": "subject",
                "balance": "balanced"
            })
            
    # 3. Embedding (8 models)
    models = ["cbramod", "reve", "biot", "labram", "luna", "eegpt", "signaljepa", "bendr"]
    for model in models:
        analyses.append({
            "type": "embedding",
            "model": model,
            "level": "subject",
            "balance": "balanced"
        })
        
    # 4. Fine-tuning (8 models)
    for model in models:
        analyses.append({
            "type": "fine_tune",
            "model": model,
            "level": "subject",
            "balance": "balanced"
        })
        
    combinations = []
    combo_id = 0
    
    for sex, age, com, med in cohorts:
        for analysis in analyses:
            combinations.append({
                "id": combo_id,
                "cohort": {
                    "sex": sex,
                    "age": age,
                    "comorbidities": com,
                    "medication": med
                },
                "analysis": analysis
            })
            combo_id += 1
            
    out_path = Path(__file__).parent / "analysis_combinations.json"
    with open(out_path, "w") as f:
        json.dump(combinations, f, indent=2)
        
    csv_path = Path(__file__).parent / "analysis_combinations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "sex", "age", "comorbidities", "medication", 
            "analysis_type", "method_or_model", "handcrafted_unit", "level", "balance"
        ])
        for c in combinations:
            cohort = c["cohort"]
            analysis = c["analysis"]
            
            atype = analysis["type"]
            method_or_model = ""
            if atype == "dim_reduction":
                method_or_model = analysis["method"]
            elif atype == "handcrafted":
                method_or_model = analysis["regression_head"]
            elif atype in ["embedding", "fine_tune"]:
                method_or_model = analysis["model"]
                
            h_unit = analysis.get("unit", "N/A")
            
            writer.writerow([
                c["id"], cohort["sex"], cohort["age"], cohort["comorbidities"], cohort["medication"],
                atype, method_or_model, h_unit, analysis["level"], analysis["balance"]
            ])
        
    print(f"Generated {len(combinations)} distinct combinations successfully at:")
    print(f"  JSON: {out_path}")
    print(f"  CSV:  {csv_path}")
    print(f"Total Cohorts: {len(cohorts)}, Analyses per cohort: {len(analyses)}")

if __name__ == "__main__":
    main()
