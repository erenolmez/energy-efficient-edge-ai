"""CPU-testable switching ablations; historical cache is never modified."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import numpy as np


def select_cache_aware(probabilities,costs,miss_costs,resident,penalty):
    q,c,miss=np.asarray(probabilities),np.asarray(costs),np.asarray(miss_costs)
    if q.ndim!=1 or c.shape!=q.shape or miss.shape!=q.shape or not np.isfinite(q).all() or (q<0).any() or (q>1).any():
        raise ValueError('Finite candidate probabilities and matching costs required')
    if not np.isfinite(c).all() or (c<=0).any() or not np.isfinite(miss).all() or (miss<0).any() or not np.isfinite(penalty) or penalty<0:
        raise ValueError('Positive execution cost, nonnegative measured miss cost and penalty required')
    if any(i<0 or i>=len(q) for i in resident):
        raise ValueError('Unknown resident candidate')
    effective=c+miss
    effective[list(resident)]=c[list(resident)]
    # Fixed normalization does not change with cache state. Penalty is validation-calibrated.
    return int(np.argmax(q-penalty*effective/(c+miss).max()))


class VerifiedHostStore:
    """Verify immutable engine bytes once; setup time/RAM must be reported.

    Keeping serialized bytes in CPU RAM is NOT keeping GPU engines resident.
    Deserialization is still charged on every device-cache miss.
    """
    def __init__(self,records):
        self.data={}
        for record in records:
            value=Path(record['engine_path']).read_bytes()
            if hashlib.sha256(value).hexdigest()!=record['engine_sha256']:
                raise RuntimeError('Engine hash mismatch')
            self.data[record['name']]=value
        self.host_bytes=sum(map(len,self.data.values()))

    def get(self,name):
        return self.data[name]


class ProfiledLRU:
    """Capacity/byte-budget cache with injectable CUDA-free synthetic tests.

    Allocation measurements must come from a separate GPU profile, not engine
    file sizes alone. No unmeasured automatic increase in resident capacity.
    """
    def __init__(self,records,capacity,byte_budget,factory,synchronize,release):
        if capacity<1 or byte_budget<=0:
            raise ValueError('Positive capacity and measured memory budget required')
        self.records,self.capacity,self.byte_budget=records,capacity,byte_budget
        self.factory,self.synchronize,self.release=factory,synchronize,release
        self.resident=OrderedDict()
        self.loads=self.evictions=0

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        if not 0<=index<len(self.records):
            raise IndexError(index)
        if index in self.resident:
            self.resident.move_to_end(index)
            return self.resident[index]
        required=self.records[index]['profiled_resident_bytes']
        if required<=0 or required>self.byte_budget:
            raise RuntimeError('Candidate exceeds separately profiled memory budget')
        self.synchronize()
        while self.resident and (len(self.resident)>=self.capacity or
            sum(self.records[i]['profiled_resident_bytes'] for i in self.resident)+required>self.byte_budget):
            _,old=self.resident.popitem(last=False)
            del old
            self.release()
            self.evictions+=1
        engine=self.factory(self.records[index])
        self.resident[index]=engine
        self.loads+=1
        return engine

    def clear(self):
        self.synchronize()
        self.resident.clear()
        self.release()


def simulate_policy(q,counts,lengths,costs,miss_costs,capacity,penalty,initial=(),forced=None):
    """Sequential validation simulation: choices affect the next cache state."""
    q,counts,lengths=np.asarray(q),np.asarray(counts),np.asarray(lengths)
    costs,miss_costs=np.asarray(costs),np.asarray(miss_costs)
    if q.ndim!=2 or counts.shape!=q.shape or lengths.shape!=(len(q),) or not len(q) or capacity<1:
        raise ValueError('Validation rows, correctness counts, positive capacity required')
    if not np.isfinite(counts).all() or not np.isfinite(lengths).all() or (lengths<=0).any() or (counts<0).any() or (counts>lengths[:,None]).any():
        raise ValueError('Invalid correctness/lengths')
    if len(initial)>capacity or len(set(initial))!=len(initial):
        raise ValueError('Invalid initial residency')
    if forced is not None and not 0<=forced<q.shape[1]:
        raise ValueError('Invalid forced reference candidate')
    resident=OrderedDict((i,None) for i in initial)
    total=correct=0.
    loads=evictions=0
    choices=[]
    for row in range(len(q)):
        # Also validates all costs/probabilities in the forced-reference case.
        selected=select_cache_aware(q[row],costs,miss_costs,list(resident),penalty)
        if forced is not None:
            selected=forced
        miss=selected not in resident
        total+=costs[selected]+(miss_costs[selected] if miss else 0.)
        correct+=counts[row,selected]
        if miss:
            if len(resident)>=capacity:
                resident.popitem(last=False);evictions+=1
            resident[selected]=None;loads+=1
        else:
            resident.move_to_end(selected)
        choices.append(selected)
    return dict(choices=choices,accuracy=100*correct/lengths.sum(),estimated_total_cost=float(total),
                estimated_cost_per_batch=float(total/len(q)),loads=loads,evictions=evictions)


def calibrate_cache_aware(q,counts,lengths,costs,miss_costs,capacity,baseline,budget_pp,initial=()):
    """Only caller-supplied validation data, including cold/warm initial state.

    Fixed65-point penalty grid; no test-data search or cached test schedule.
    """
    if not np.isfinite(budget_pp) or budget_pp<0:
        raise ValueError('Nonnegative accuracy allowance required')
    reference=simulate_policy(q,counts,lengths,costs,miss_costs,capacity,0.,initial,forced=baseline)
    best=dict(fallback=True,baseline_index=int(baseline),penalty=0.,**reference)
    minimum=reference['accuracy']-budget_pp
    for penalty in np.r_[0.,np.geomspace(.0001,100.,64)]:
        candidate=simulate_policy(q,counts,lengths,costs,miss_costs,capacity,float(penalty),initial)
        if candidate['accuracy']+1e-10>=minimum and candidate['estimated_total_cost']<best['estimated_total_cost']:
            best=dict(fallback=False,baseline_index=int(baseline),penalty=float(penalty),**candidate)
    best.pop('choices')  # Never use a cached calibration schedule at runtime.
    return best
