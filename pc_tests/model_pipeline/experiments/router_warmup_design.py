"""Frozen small-router warm-up design; pure CPU schedule/validation helpers."""
import math


def slots():
    orders=((42,45,48),(45,48,42),(48,42,45),(48,45,42),(45,42,48),(42,48,45))
    rows=[]
    for block, seeds in enumerate(orders):
        conditions=('cold','warm') if block%2==0 else ('warm','cold')
        for position, seed in enumerate(seeds):
            for condition in conditions:
                rows.append(dict(id=f'b{block+1}_p{position+1}_s{seed}_{condition}',
                                 block=block+1,position=position+1,seed=seed,condition=condition))
    return rows


def ready(temperature, reason):
    return isinstance(temperature,(int,float)) and math.isfinite(temperature) and temperature<=70 and reason is not None and not (reason&32)


def require_energy(metrics):
    value=metrics.get('gpu_energy_j')
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise RuntimeError('Unavailable/nonpositive GPU fit energy; preserve and stop')
    return metrics
