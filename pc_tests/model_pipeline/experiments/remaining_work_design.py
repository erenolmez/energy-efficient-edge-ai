"""Frozen finite follow-up schedules, importable without CUDA."""
import itertools


def graph_windows():
    modes=list(itertools.permutations(('cpu_dispatch','gpu_conditional','cuda_graph')))
    strata=list(itertools.product(('host','gpu'),('per_image_cpu','deferred_gpu')))
    rows=[]
    for repeat,order in enumerate(modes):
        rotated=strata[repeat%4:]+strata[:repeat%4]
        for residence,egress in rotated:
            rows.extend(dict(repeat=repeat,mode=mode,residence=residence,egress=egress) for mode in order)
    return rows


def plateau_windows():
    strata=list(itertools.product(('resnet101__p0__fp16','efficientnet_b0__p0__fp16'),('gpu_resident','host_to_logits')))
    rows=[]
    for repeat,order in enumerate(((32,64,128),(64,128,32),(128,32,64))):
        for name,boundary in strata[repeat:]+strata[:repeat]:
            rows.extend(dict(repeat=repeat,name=name,boundary=boundary,batch=b) for b in order)
    return rows
