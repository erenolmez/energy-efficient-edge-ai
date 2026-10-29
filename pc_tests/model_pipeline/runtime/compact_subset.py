"""Evaluate only available independent heads; shared networks retain full cost."""
import numpy as np
from pc_tests.model_pipeline.routers.correctness.compact_cpu import features,thumbnails

def predict(model,rgb,columns):
    columns=np.asarray(columns,dtype=np.int64)
    if columns.ndim!=1 or not len(columns) or (columns<0).any() or (columns>=model.candidates).any():
        raise ValueError('Valid candidate columns required')
    x=thumbnails(rgb).reshape(len(rgb),192) if model.method=='thumbnail_mlp16' else features(rgb)
    if model.method=='logistic':
        z=model.scaler.transform(x)
        result=np.column_stack([np.full(len(x),h) if isinstance(h,float) else h.predict_proba(z)[:,1] for h in [model.heads[i] for i in columns]])
    elif model.method=='boosted_d2':
        result=np.column_stack([np.full(len(x),h) if isinstance(h,float) else h.predict(x) for h in [model.heads[i] for i in columns]])
    else:
        result=model.predict_inputs(x)[:,columns]
    result=np.asarray(result,dtype=np.float32).clip(0,1)
    if not np.isfinite(result).all(): raise RuntimeError('Nonfinite compact subset output')
    return result
