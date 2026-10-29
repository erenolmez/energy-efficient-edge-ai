"""Fresh CPU batch-one descriptors and four compact correctness families.

No file access, CUDA initialization or test-label-based model selection.
"""
import numpy as np

METHODS=('logistic','tree_d3','thumbnail_mlp16','boosted_d2')
FEATURE_NAMES=('r_mean','g_mean','b_mean','r_std','g_std','b_std',
               'gray_mean','gray_std','gradient_mean','gradient_std','edge_density','saturation_proxy')


def thumbnails(rgb):
    values=np.asarray(rgb)
    if values.ndim!=4 or values.shape[1:]!=(3,128,128) or values.dtype!=np.uint8 or not len(values):
        raise ValueError('Nonempty uint8 NCHW RGB128 required')
    return values.reshape(len(values),3,8,16,8,16).mean(axis=(3,5),dtype=np.float32)/255.


def features(rgb):
    t=thumbnails(rgb)
    gray=.299*t[:,0]+.587*t[:,1]+.114*t[:,2]
    dx=np.diff(gray,axis=2)[:,:-1,:]
    dy=np.diff(gray,axis=1)[:,:,:-1]
    gradient=np.sqrt(dx*dx+dy*dy)
    result=np.column_stack((t.mean(axis=(2,3)),t.std(axis=(2,3)),
        gray.mean(axis=(1,2)),gray.std(axis=(1,2)),gradient.mean(axis=(1,2)),gradient.std(axis=(1,2)),
        (gradient>.1).mean(axis=(1,2)),(t.max(axis=1)-t.min(axis=1)).mean(axis=(1,2))))
    return result.astype(np.float32)


class CompactCPU:
    """Predict per-candidate correctness, not CIFAR classes or a scalar rank."""
    def __init__(self,method,candidates,model=None,scaler=None,heads=None):
        self.method,self.candidates,self.model,self.scaler,self.heads=method,candidates,model,scaler,heads

    def predict_inputs(self,x):
        x=np.asarray(x,dtype=np.float32)
        if self.method=='logistic':
            z=self.scaler.transform(x)
            out=np.column_stack([np.full(len(x),head) if isinstance(head,float) else head.predict_proba(z)[:,1] for head in self.heads])
        elif self.method=='boosted_d2':
            out=np.column_stack([np.full(len(x),head) if isinstance(head,float) else head.predict(x) for head in self.heads])
        elif self.method=='tree_d3':
            out=self.model.predict(x)
        else:
            out=self.model.predict_proba(x)
        out=np.asarray(out,dtype=np.float32).reshape(len(x),self.candidates).clip(0,1)
        if not np.isfinite(out).all():
            raise RuntimeError('Nonfinite CPU correctness output')
        return out

    def predict_rgb(self,rgb):
        # This entire function belongs inside the deployment wall/energy window.
        x=thumbnails(rgb).reshape(len(rgb),192) if self.method=='thumbnail_mlp16' else features(rgb)
        return self.predict_inputs(x)


def fit(method,x,y,seed=42,boost_trees=16,mlp_epochs=20):
    from threadpoolctl import threadpool_limits
    x,y=np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32)
    if method not in METHODS or x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x):
        raise ValueError('Supported method and fit-only input/target matrices required')
    if not np.isfinite(x).all() or not np.isin(y,(0,1)).all():
        raise ValueError('Finite inputs and binary candidate correctness required')
    with threadpool_limits(limits=1):
        if method=='tree_d3':
            from sklearn.tree import DecisionTreeRegressor
            estimator=DecisionTreeRegressor(max_depth=3,min_samples_leaf=20,random_state=seed).fit(x,y)
            return CompactCPU(method,y.shape[1],model=estimator)
        if method=='logistic':
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            scaler=StandardScaler().fit(x)
            z=scaler.transform(x)
            heads=[float(column[0]) if np.unique(column).size==1 else LogisticRegression(C=.25,max_iter=2000,random_state=seed).fit(z,column) for column in y.T]
            return CompactCPU(method,y.shape[1],scaler=scaler,heads=heads)
        if method=='boosted_d2':
            from xgboost import XGBRegressor
            heads=[float(column[0]) if np.unique(column).size==1 else XGBRegressor(n_estimators=boost_trees,max_depth=2,
                learning_rate=.04,min_child_weight=20,reg_lambda=15,reg_alpha=.2,subsample=.85,colsample_bytree=.85,
                objective='reg:squarederror',tree_method='hist',device='cpu',n_jobs=1,random_state=seed).fit(x,column) for column in y.T]
            return CompactCPU(method,y.shape[1],heads=heads)
        from sklearn.neural_network import MLPClassifier
        model=MLPClassifier(hidden_layer_sizes=(16,),activation='relu',solver='adam',alpha=.0001,batch_size=64,
            learning_rate_init=.001,max_iter=mlp_epochs,early_stopping=False,tol=0.,n_iter_no_change=mlp_epochs+1,
            random_state=seed).fit(x,y.astype(int))
        return CompactCPU(method,y.shape[1],model=model)
