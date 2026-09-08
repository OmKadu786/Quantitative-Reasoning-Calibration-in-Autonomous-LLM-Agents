import json
import glob
import os

def generate_master_dump():
    files = sorted(glob.glob('logs/*/*/summary_*.json'))
    master_data = []

    for f in files:
        try:
            with open(f, 'r') as infile:
                data = json.load(infile)
            
            # f is like logs/claude/ecg/summary_ecg_c1_claude.json
            parts = f.split('/')
            model_fam = parts[1]
            dataset = parts[2]
            filename = parts[3]
            condition = filename.split('_')[2]
            
            cond_data = {
                "model": model_fam,
                "dataset": dataset,
                "condition": condition,
                "runs": []
            }
            
            for run in data:
                run_idx = run.get('run_index')
                trajectory = []
                for it in run.get('trajectory', []):
                    c = it['calibration']
                    m = it['actual_means']
                    
                    sig_acc = c.get('signal_agent_directional_acc')
                    pct_sig = c.get('pct_detectable_signal_metrics')
                    
                    trajectory.append({
                        "iteration": it['iteration'],
                        "actual_macro_f1": round(m.get('macro_f1', 0), 4),
                        "mace": round(c.get('mace', 0), 4),
                        "rmace_mean": round(c.get('mean_relative_mace', 0), 2),
                        "rmace_median": round(c.get('median_relative_mace', 0), 2),
                        "pooled_accuracy": round(c.get('agent_directional_accuracy_rate', 0), 2),
                        "signal_accuracy": round(sig_acc, 2) if sig_acc is not None else None,
                        "signal_percent": round(pct_sig, 2) if pct_sig is not None else 0,
                        "overconfidence": round(c.get('overconfidence_rate', 0), 2)
                    })
                
                cond_data["runs"].append({
                    "run_index": run_idx,
                    "trajectory": trajectory
                })
                
            master_data.append(cond_data)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    output_path = 'master_results_dump.json'
    with open(output_path, 'w') as outfile:
        json.dump(master_data, outfile, indent=2)
    print(f"✅ Created {output_path} with {len(master_data)} completed conditions.")

if __name__ == '__main__':
    generate_master_dump()
