"""CPU index partition for independent per-image model decisions."""
import numpy as np

def groups(choices,candidates):
    choices=np.asarray(choices)
    if choices.ndim!=1 or not len(choices) or not np.issubdtype(choices.dtype,np.integer) or ((choices<0)|(choices>=candidates)).any():
        raise ValueError('Valid nonempty integer modelchoices required')
    return [(int(i),np.flatnonzero(choices==i)) for i in np.unique(choices)]
