"""Reconcile recorded asynchronous snapshots without deleting edge evidence."""
from datetime import datetime

MAX_EDGE_SECONDS = .025


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def reconcile(row, samples):
    """Require recoverable snapshot count/flags; conservatively retain all rows.

    Labels are captured before a multi-query NVML read finishes. A sample may
    append between end reading and snapshot capture, or after that capture.
    This is NOT an exact reconstruction of timer/snapshot membership.
    """
    count=row['timed_telemetry_samples']
    assert samples and count>0
    start,end=timestamp(row['start_reading']['utc']),timestamp(row['end_reading']['utc'])
    assert end>start
    times=[timestamp(s['utc']) for s in samples]
    assert all(b>a for a,b in zip(times,times[1:]))
    assert all(start-MAX_EDGE_SECONDS<=t<=end+MAX_EDGE_SECONDS for t in times), 'Unexpected phase spill beyond narrow edge tolerance'
    outside=[i for i,t in enumerate(times) if t<start or t>end]
    assert len(samples)>=count, 'Saved snapshot samples missing from phase-labelled telemetry'
    assert len(samples)-count<=1, 'More than one unexplained sample-count difference'
    candidates=[samples] if len(samples)==count else []
    if len(samples)!=count:
        # Only a timestamp-supported edge sample may be outside the snapshot.
        for edge in (0,len(samples)-1):
            if edge in outside: candidates.append(samples[1:] if edge==0 else samples[:-1])
    def flags(values,mask):
        return sum(bool(v['clock_reasons']&mask) for v in [row['start_reading'],*values,row['end_reading']])
    compatible=[c for c in candidates if flags(c,4)==row['software_powercap_samples'] and flags(c,32)==row['software_thermal_samples']]
    assert compatible, 'Snapshot count/flags cannot be reconciled from supported edge candidates'
    conservative_thermal=flags(samples,32)
    assert conservative_thermal==0, 'Thermal evidence in conservative envelope; do not discard edge flags'
    assert max([times[0]-start,end-times[-1],*[b-a for a,b in zip(times,times[1:])]])<2
    return dict(index=row['index'],recorded_snapshot_samples=count,phase_labelled_samples=len(samples),
        snapshot_membership_compatible=True,membership_not_exactly_reconstructed=True,
        edge_samples=[dict(utc=samples[i]['utc'],offset_from_nearest_boundary_s=min(abs(times[i]-start),abs(times[i]-end)),
            clock_reasons=samples[i]['clock_reasons']) for i in outside],
        recorded_powercap_samples=row['software_powercap_samples'],conservative_powercap_samples=flags(samples,4),
        conservative_thermal_samples=conservative_thermal,
        max_edge_tolerance_seconds=MAX_EDGE_SECONDS)
