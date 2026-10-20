"""CPU-only export smoke checks for router implementations, not learned results."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'router_benchmark'))
from router_components import TinyCNN
from gpu_router_models import TreeRouter, LogisticRouter, XGBRouter, AggregatedTreeRouter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    features = rng.normal(size=(200,13)).astype('float32')
    targets = features[:,0] + features[:,8] ** 2
    tree = DecisionTreeRegressor(max_depth=3,random_state=42).fit(features,targets)
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression().fit(scaler.transform(features),targets>np.median(targets))
    xgb = XGBRegressor(n_estimators=12,max_depth=2,n_jobs=2,random_state=42).fit(features,targets)
    fixtures = [('tree',TreeRouter(tree.tree_),(1,13)),
                ('logistic',LogisticRouter(scaler,classifier),(1,13)),
                ('xgboost',XGBRouter(xgb.get_booster()),(1,13)),
                ('tinycnn',TinyCNN(),(1,3,32,32))]
    fixtures += [(f'aggregate_{percentile}',AggregatedTreeRouter(TreeRouter(tree.tree_),32,percentile),(32,13))
                 for percentile in (10,25,50,70,80,90,None)]
    records = []
    for name, model, shape in fixtures:
        model.eval()
        path = args.output_dir / f'{name}.onnx'
        torch.onnx.export(model,torch.zeros(shape),path,input_names=['input'],output_names=['difficulty'],
                          opset_version=17,dynamo=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads=2
        session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        maximum_error = 0.
        with torch.inference_mode():
            for _ in range(100):
                values = rng.normal(size=shape).astype('float32')
                expected = model(torch.from_numpy(values)).numpy()
                actual = session.run(None,{'input':values})[0]
                maximum_error=max(maximum_error,float(np.max(np.abs(actual-expected))))
                if not np.isfinite(actual).all() or not np.allclose(actual,expected,atol=1e-5,rtol=1e-4):
                    raise RuntimeError(f'{name}: ONNX parity failed: {maximum_error}')
        records.append({'name':name,'input_shape':shape,'checks':100,'maximum_error':maximum_error,'status':'complete'})
        print(f'{name}: CPU ONNX parity passed',flush=True)
    (args.output_dir/'preflight.json').write_text(json.dumps({'scope':'Synthetic CPU export/ORT parity only; not TensorRT GPU validation or trained router performance',
                                                            'results':records},indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
