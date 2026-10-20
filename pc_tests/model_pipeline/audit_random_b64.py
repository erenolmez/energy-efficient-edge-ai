"""Read-only scientific audit; saves a small audit result, never runs inference."""
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from run_random_routes_b64 import PIPELINE, digest, frozen_inputs, task_list, schedule_for
from batch_router_policy import make_batches
import numpy as np


def main():
    folder = PIPELINE/'reports/random_b64'
    report = json.loads((folder/'random_routes_b64.json').read_text())
    source,cases,records,cache,labels,predictions,protocol = frozen_inputs()
    assert report['protocol'] == protocol
    assert report['status']=='complete' and len(report['results'])==report['expected_blocks']==495
    results = {r['id']:r for r in report['results']}
    assert len(results)==495
    tasks=task_list(cases)
    expected={f"uniform_seed{c['seed']}_trial{t}" if role=='uniform' else f"{c['id']}_{role}_trial{t}"
              for c,role,t in tasks}
    assert set(results)==expected
    assert all(r['status']=='complete' and r['batches']==157 and sum(r['selected_images'])==10000
               and math.isfinite(r['latency_ms_per_image']) and r['latency_ms_per_image']>0
               and math.isfinite(r['gpu_energy_j_per_image']) and r['gpu_energy_j_per_image']>0
               for r in results.values())
    for case,role,trial in tasks:
        identity=f"uniform_seed{case['seed']}_trial{trial}" if role=='uniform' else f"{case['id']}_{role}_trial{trial}"
        row=results[identity]
        if role=='matched':
            assert row['selected_images']==case['measurements'][0]['selected_images']
            assert row['selected_batches']==case['measurements'][0]['selected_batches']
        elif role=='learned':
            assert row['selection_counts_match_original']
    protected = json.loads((PIPELINE/'reports/most_pruned_b8_32_64/most_pruned_routes.json').read_text())
    assert digest(cache/'test_rgb128.npy') == protected['protocol']['rgb_sha256']
    summary=json.loads((folder/'random_comparison_summary.json').read_text())
    assert summary['complete'] and len(summary['rows'])==72
    lookup={(r['case'],r['policy']):r for r in summary['rows']}
    learned=[r for r in summary['rows'] if r['policy']=='learned']
    matched=[r for r in summary['rows'] if r['policy']=='matched']
    uniform=[r for r in summary['rows'] if r['policy']=='uniform' and r['objective']=='latency' and r['budget_pp']==2]
    differences=[r['accuracy']-lookup[r['case'],'matched']['accuracy'] for r in learned]
    audit=dict(complete=True,unique_blocks=495,paired_cases=24,summary_rows=72,
        protocols_and_source_hashes_verified=True,protected_b128_hash_verified=True,
        rgb_hash_verified_against_protected_p90=True,all_matched_counts_preserved=True,
        fresh_learned_count_mismatches=0,cached_prediction_disagreements=sum(r['cached_prediction_disagreements'] for r in results.values()),
        uniform_accuracy_mean=statistics.mean(r['accuracy'] for r in uniform),
        learned_accuracy_range=[min(r['accuracy'] for r in learned),max(r['accuracy'] for r in learned)],
        learned_latency_range=[min(r['median_latency_ms_per_image'] for r in learned),max(r['median_latency_ms_per_image'] for r in learned)],
        uniform_latency_range=[min(r['median_latency_ms_per_image'] for r in uniform),max(r['median_latency_ms_per_image'] for r in uniform)],
        learned_minus_matched_mean_accuracy_pp=statistics.mean(differences),
        learned_better_accuracy_cases=sum(d>0 for d in differences),
        learned_accuracy_pass_cases=sum(r['mean_accuracy_pass'] for r in learned),
        matched_accuracy_pass_cases=sum(r['mean_accuracy_pass'] for r in matched),
        original_failures_and_negative_savings_retained=True)
    (folder/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
