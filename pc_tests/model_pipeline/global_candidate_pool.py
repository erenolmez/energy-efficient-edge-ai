"""All-architecture candidate identities and bounded runtime residency."""
from collections import OrderedDict


def global_candidates(records):
    """Keep every measured variant; baseline uses validation accuracy only."""
    if len(records) != 213 or len({r['name'] for r in records}) != 213:
        raise ValueError('All 213 unique measured configurations are required')
    if len({r['architecture'] for r in records}) != 8:
        raise ValueError('All eight architectures are required')
    if any(r.get('status') != 'complete' for r in records):
        raise ValueError('Incomplete static candidate')
    candidates = sorted(records, key=lambda r: r['name'])
    references = [r for r in candidates if r['prune'] == 0 and r['precision'] == 'fp32']
    baseline = min(references, key=lambda r: (-r['validation_accuracy'], r['name']))
    return candidates, baseline['name']


class EngineCache:
    """Every candidate is selectable; only capacity engines are resident.

    Access must happen inside the measured dispatch. Cache misses, verification,
    deserialization, allocation, synchronization and eviction are not hidden.
    Dependencies are injectable so cache behavior can be tested without CUDA.
    """
    def __init__(self, records, capacity, factory, synchronize, release):
        if capacity < 1:
            raise ValueError('Positive resident capacity required')
        self.records = records
        self.capacity = capacity
        self.factory, self.synchronize, self.release = factory, synchronize, release
        self.resident = OrderedDict()
        self.loads = self.evictions = 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if not 0 <= index < len(self.records):
            raise IndexError(index)
        if index in self.resident:
            self.resident.move_to_end(index)
            return self.resident[index]
        self.synchronize()
        if len(self.resident) >= self.capacity:
            _, old = self.resident.popitem(last=False)
            del old
            self.release()
            self.evictions += 1
        engine = self.factory(self.records[index])
        self.resident[index] = engine
        self.loads += 1
        return engine

    def clear(self):
        self.synchronize()
        self.resident.clear()
        self.release()
