"""Reproducible input-only router comparison against the fixed FP32 p0 model.

GPU training; serial tree inference; disjoint fit/calibration/evaluation roles.
Candidate correctness comes from cached FP32 predictions. Jetson costs are
estimates for selected candidates, never claimed as measured router-system costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
sys.modules["numexpr"] = None
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torchvision.datasets import CIFAR100
from torchvision import transforms
from xgboost import XGBClassifier, XGBRegressor

from features import extract_features, extract_enhanced_features
from router_components import MobileNetEmbedder, TinyCNN, split_positions
from router_features_v2 import FEATURE_SETS, extract_feature_vector
from routing_policy_v2 import calibrate_utility, select_utility

ROOT = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def xgb(seed, depth=3, trees=250, classifier=False):
    cls = XGBClassifier if classifier else XGBRegressor
    return cls(n_estimators=trees, max_depth=depth, learning_rate=0.04,
               min_child_weight=15, reg_lambda=10, reg_alpha=0.1,
               subsample=0.85, colsample_bytree=0.85,
               objective="binary:logistic" if classifier else "reg:squarederror",
               tree_method="hist", device="cuda", n_jobs=4, random_state=seed)


def scalar_policy(scores, correct, costs, p0, drop):
    order = np.argsort(costs, kind="stable")
    minimum = float(correct[:, p0].mean() - drop)
    best = (float(costs[p0]), -float(correct[:, p0].mean()))
    policy = {"kind": "scalar", "fallback": True, "bias": 0.0,
              "calibration_accuracy": -best[1], "calibration_cost": best[0],
              "minimum_accuracy": minimum}
    for bias in np.linspace(-1, 1, 801):
        thresholds = (np.arange(len(costs)-1) + 0.5)/(len(costs)-1) - bias
        choices = order[np.digitize(scores, thresholds)]
        acc = correct[np.arange(len(choices)), choices].mean()
        key = (costs[choices].mean(), -acc)
        if acc + 1e-12 >= minimum and key < best:
            best = key
            policy.update(fallback=False, bias=float(bias),
                          calibration_accuracy=float(acc), calibration_cost=float(key[0]))
    return policy


def select_scalar(scores, costs, policy, p0):
    if policy["fallback"]:
        return np.full(len(scores), p0, dtype=int)
    thresholds = (np.arange(len(costs)-1)+0.5)/(len(costs)-1) - policy["bias"]
    return np.argsort(costs, kind="stable")[np.digitize(scores, thresholds)]


def conditional_inputs(x, number_models):
    return np.hstack([np.repeat(x, number_models, axis=0),
                      np.tile(np.eye(number_models, dtype=np.float32), (len(x), 1))])


def benchmark_one(images, predictor, device, count):
    for im in images[:20]:
        predictor(im)
    sync(device)
    values = []
    for im in images[:count]:
        sync(device)
        start = time.perf_counter()
        predictor(im)
        sync(device)
        values.append(1000 * (time.perf_counter()-start))
    return {"mean_ms": float(np.mean(values)),
            "p50_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "samples": len(values)}


def feature_extractor(name):
    if name == "basic":
        return lambda im: np.array(list(extract_features(im).values()), np.float32)
    if name == "enhanced":
        return lambda im: np.array(list(extract_enhanced_features(im).values()), np.float32)
    return lambda im: extract_feature_vector(im, name)


def features_cached(images, name, out):
    # Inputs are always the complete ordered CIFAR-100 test split (validated below).
    path = out / f"features_{name}.npy"
    extract = feature_extractor(name)
    if path.exists():
        x = np.load(path)
        if x.shape[0] != len(images) or not np.allclose(x[0], extract(images[0])):
            raise ValueError(f"Stale feature cache: {path}")
    else:
        x = np.stack([extract(im) for im in images])
        np.save(path, x)
    return x, extract


class DirectRouter(nn.Module):
    def __init__(self, outputs, mlp=False, size=16):
        super().__init__()
        if mlp:
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(3*size*size, 96),
                                     nn.ReLU(), nn.Dropout(0.15), nn.Linear(96, outputs))
        else:
            self.net = TinyCNN().net
            self.net[-1] = nn.Linear(48, outputs)

    def forward(self, x):
        return self.net(x)


def image_tensor(images, size):
    x = torch.from_numpy(np.asarray(images).copy()).permute(0,3,1,2).float()/255
    if size != 32:
        x = F.interpolate(x, size=(size,size), mode="bilinear", align_corners=False,
                          antialias=True)
    return (x - torch.tensor([.5071,.4867,.4408])[None,:,None,None]) / torch.tensor(
        [.2675,.2565,.2761])[None,:,None,None]


def train_neural(x, y, fit, labels, seed, device, epochs, mlp=False, scalar=False,
                 weights=None):
    seed_all(seed)
    train, valid = train_test_split(fit, test_size=.2, random_state=107,
                                   stratify=labels[fit])
    model = DirectRouter(y.shape[1], mlp=mlp, size=x.shape[-1]).to(device)
    xx = x.to(device)
    yy = torch.as_tensor(y, dtype=torch.float32, device=device)
    ww = torch.ones(len(y), device=device) if weights is None else torch.as_tensor(
        weights, dtype=torch.float32, device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.005)
    best, state, best_epoch = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for batch in np.array_split(np.random.permutation(train), max(1, int(np.ceil(len(train)/128)))):
            pred = model(xx[batch])
            loss = ((pred-yy[batch]).square().mean(1)*ww[batch]).mean() if scalar else F.binary_cross_entropy_with_logits(pred, yy[batch])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        model.eval()
        with torch.inference_mode():
            pred = model(xx[valid])
            loss = (pred-yy[valid]).square().mean().item() if scalar else F.binary_cross_entropy_with_logits(pred, yy[valid]).item()
        if loss < best:
            best, best_epoch = loss, epoch+1
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        scores = torch.cat([model(b) for b in xx.split(256)])
        if not scalar:
            scores = scores.sigmoid()
        scores = scores.cpu().numpy()
    del xx, yy, ww
    return model, scores.squeeze(1) if scalar else scores, best_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--latency-samples", type=int, default=200)
    p.add_argument("--max-accuracy-drop", type=float, default=.01)
    p.add_argument("--methods", default="all", help="Comma-separated names or all")
    p.add_argument("--objective", choices=("energy", "latency"), required=True,
                   help="Run energy and latency as separate experiments.")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this experiment")
    device = "cuda"
    active_objectives = (args.objective,)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(ROOT/"artifacts/torch_cache"))
    data = pd.read_csv(ROOT/"artifacts/features.csv")
    data = data[data["split"] == "test"].reset_index(drop=True)
    base = CIFAR100(root=args.data_dir, train=False, download=False)
    images = base.data
    labels = np.array(base.targets)
    expected_ids = [f"test:{i:05d}" for i in range(len(images))]
    if data.image_id.tolist() != expected_ids or not np.array_equal(data.true_label, labels):
        raise ValueError("Image IDs, order, or labels differ from CIFAR-100")
    predictions = pd.read_csv(ROOT/"artifacts/predictions_all_fp32.csv")
    if predictions.duplicated(["image_id", "model_id"]).any():
        raise ValueError("Duplicate predictions")
    models = [f"fp32_p{p}" for p in range(0,100,10)]
    p0 = models.index("fp32_p0")
    table = predictions.assign(correct=predictions.true_label == predictions.predicted_label).pivot(
        index="image_id", columns="model_id", values="correct").reindex(index=expected_ids, columns=models)
    if table.isna().any().any():
        raise ValueError("Incomplete predictions")
    correct = table.to_numpy(bool)
    for m in models:
        yy = predictions[predictions.model_id == m].set_index("image_id").loc[expected_ids, "true_label"].to_numpy()
        if not np.array_equal(yy, labels):
            raise ValueError(f"Labels disagree for {m}")
    fit, calibration, evaluation = split_positions(data, args.split_seed)
    split = {name: pos.tolist() for name,pos in [("fit",fit),("calibration",calibration),("evaluation",evaluation)]}
    split_hash = hashlib.sha256(np.asarray(evaluation,dtype=np.int64).tobytes()).hexdigest()
    (out/"split_indices.json").write_text(json.dumps({**split, "evaluation_sha256":split_hash},indent=2))
    costs = {}
    for objective in ("energy", "latency"):
        c = pd.read_csv(ROOT/f"artifacts/costs_{objective}_all_fp32.csv").set_index("model_id")
        costs[objective] = c.loc[models,"cost"].to_numpy(float)
    print(f"CUDA {torch.cuda.get_device_name(0)}; fit/cal/eval {len(fit)}/{len(calibration)}/{len(evaluation)}; p0 accuracy {correct[evaluation,p0].mean():.2%}", flush=True)
    features = {}
    for name in ("basic", "enhanced", "compact", "texture"):
        features[name] = features_cached(images, name, out)
        print(f"Feature set {name}: {features[name][0].shape[1]} values", flush=True)
    # A frozen model already cached locally; embeddings are reused across seeds.
    embed_path = out/"embeddings.npy"
    embedder = MobileNetEmbedder().to(device).eval()
    embed_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((96,96)),
        transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    if embed_path.exists():
        embeddings = np.load(embed_path)
    else:
        blocks=[]
        with torch.inference_mode():
            for start in range(0,len(images),128):
                batch = torch.stack([embed_transform(im) for im in images[start:start+128]]).to(device)
                blocks.append(embedder(batch).cpu().numpy())
        embeddings = np.concatenate(blocks)
        np.save(embed_path, embeddings)
    def extract_embedding(im):
        with torch.inference_mode():
            return embedder(embed_transform(im)[None].to(device)).cpu().numpy()[0]
    features["embedding"] = (embeddings, extract_embedding)
    tensors = {size:image_tensor(images,size) for size in (8,16,32)}

    method_specs = [
        ("basic_xgb_scalar", "basic", "scalar", 4,400,"balanced"),
        ("enhanced_xgb_scalar", "enhanced", "scalar",4,400,"balanced"),
        ("compact_xgb_shallow", "compact", "scalar",2,200,"uniform"),
        ("texture_xgb_regularized", "texture", "scalar",3,250,"uniform"),
        ("texture_xgb_balanced", "texture", "scalar",3,250,"balanced"),
        ("texture_top32_xgb", "texture", "top32",3,250,"uniform"),
        ("enhanced_extra_trees_serial", "enhanced", "extra",0,128,"uniform"),
        ("enhanced_random_forest_serial", "enhanced", "forest",0,128,"uniform"),
        ("compact_xgb_correctness", "compact", "conditional",3,300,"uniform"),
        ("texture_xgb_correctness", "texture", "conditional",4,400,"uniform"),
        ("texture_xgb_advantage", "texture", "advantage",3,300,"uniform"),
        ("texture_ridge_advantage", "texture", "ridge",0,0,"uniform"),
        ("compact_knn_correctness", "compact", "knn",0,0,"uniform"),
        ("frozen_mobilenet_xgb_scalar", "embedding", "scalar",4,400,"balanced"),
        ("frozen_mobilenet_xgb_correctness", "embedding", "conditional",3,300,"uniform"),
        ("tinycnn8_scalar", "8", "cnn_scalar",0,0,"balanced"),
        ("tinycnn16_scalar", "16", "cnn_scalar",0,0,"balanced"),
        ("tinycnn32_scalar", "32", "cnn_scalar",0,0,"balanced"),
        ("tinycnn8_correctness", "8", "cnn",0,0,"uniform"),
        ("tinycnn16_correctness", "16", "cnn",0,0,"uniform"),
        ("tinycnn32_correctness", "32", "cnn",0,0,"uniform"),
        ("thumbnail_mlp_correctness", "8", "mlp",0,0,"uniform"),
    ]
    if args.methods != "all":
        wanted=set(args.methods.split(","))
        if wanted-set(s[0] for s in method_specs):
            raise ValueError("Unknown method requested")
        method_specs=[s for s in method_specs if s[0] in wanted]
    metadata = {"baseline":"fp32_p0", "split_seed":args.split_seed, "training_seeds":args.seeds,
        "split":{k:len(v) for k,v in split.items()}, "evaluation_sha256":split_hash,
        "epochs":args.epochs,"max_calibration_accuracy_drop":args.max_accuracy_drop,
        "methods":method_specs, "objective":args.objective,
        "gpu":torch.cuda.get_device_name(0), "torch":torch.__version__,
        "latency_scope":"PC batch1 RGB array -> features/thumbnail -> prediction -> routing decision; no file I/O or selected network",
        "energy_scope":"Prior Jetson power x E2E latency cost table; router energy NOT measured",
        "reuse_warning":"Same 2000 images used in earlier exploratory runs; not a fresh confirmatory test set"}
    (out/"run_config.json").write_text(json.dumps(metadata,indent=2))
    allrows=[]; allcounts=[]
    # Shared pairing lets both objectives reuse correctness/advantage models.
    for seed in map(int,args.seeds.split(",")):
        for name, feat, kind, depth, trees, weighting in method_specs:
            objectives = active_objectives if kind in ("scalar","top32","cnn_scalar","extra","forest") else ("both",)
            for trained_objective in objectives:
                key=f"{name}_s{seed}_{trained_objective}"
                result_file=out/f"{key}.json"
                if result_file.exists():
                    prior=json.loads(result_file.read_text())
                    allrows.extend(prior["rows"]); allcounts.extend(prior["counts"])
                    print(f"Resume {key}",flush=True)
                    continue
                start=time.perf_counter(); seed_all(seed)
                scalar = trained_objective != "both"
                y = correct.astype(np.float32)
                if scalar:
                    c=costs[trained_objective]
                    order=np.argsort(c,kind="stable")
                    choices=np.where(correct[:,order].any(1),correct[:,order].argmax(1),int(np.where(order==p0)[0][0]))
                    y=choices.astype(np.float32)/(len(models)-1)
                    freq=np.bincount(choices[fit],minlength=len(models))
                    weights=np.minimum(len(fit)/(len(models)*np.maximum(freq[choices],1)),10).astype(np.float32) if weighting=="balanced" else np.ones(len(y),np.float32)
                selected_cols=None; best_epoch=None
                if kind in ("cnn", "cnn_scalar", "mlp"):
                    size=int(feat)
                    model,scores,best_epoch=train_neural(tensors[size],y[:,None] if scalar else y,
                        fit,labels,seed,device,args.epochs,mlp=kind=="mlp",scalar=scalar,
                        weights=weights if scalar else None)
                    def predictor(im, model=model,size=size,scalar=scalar):
                        with torch.inference_mode():
                            pred=model(image_tensor(im[None],size).to(device))
                            return pred.cpu().numpy()[0,0] if scalar else pred.sigmoid().cpu().numpy()[0]
                    latency_device=device
                    torch.save({"state_dict":model.state_dict(),"size":size,"kind":kind,
                                "outputs":1 if scalar else len(models),"models":models},out/f"{key}.pth")
                else:
                    x,extract=features[feat]
                    if kind=="top32":
                        selector=xgb(seed,3,200)
                        selector.fit(x[fit],y[fit]); selector.set_params(device="cpu",n_jobs=1)
                        selected_cols=np.argsort(selector.feature_importances_)[-32:]
                        x=x[:,selected_cols]
                    if kind in ("conditional","advantage"):
                        model=xgb(seed,depth,trees,classifier=kind=="conditional")
                        targets=correct.astype(np.float32) if kind=="conditional" else correct.astype(np.float32)-correct[:,[p0]]
                        model.fit(conditional_inputs(x[fit],len(models)), targets[fit].ravel())
                        model.set_params(device="cpu",n_jobs=1)
                        predict = (lambda xx:model.predict_proba(xx)[:,1]) if kind=="conditional" else model.predict
                        scores=predict(conditional_inputs(x,len(models))).reshape(-1,len(models))
                        if kind=="advantage": scores[:,p0]=0
                    elif kind in ("ridge","knn"):
                        model=make_pipeline(StandardScaler(),Ridge(alpha=1000) if kind=="ridge" else KNeighborsRegressor(n_neighbors=100,weights="distance",n_jobs=1))
                        model.fit(x[fit],y[fit]-y[fit,[p0]][:,None] if kind=="ridge" else y[fit])
                        scores=model.predict(x)
                    else:
                        if kind=="extra": model=ExtraTreesRegressor(n_estimators=128,max_depth=12,min_samples_leaf=15,n_jobs=4,random_state=seed)
                        elif kind=="forest": model=RandomForestRegressor(n_estimators=128,max_depth=12,min_samples_leaf=15,n_jobs=4,random_state=seed)
                        else: model=xgb(seed,depth,trees)
                        # Legacy controls retain their original regressor hyperparameters.
                        if name in ("basic_xgb_scalar","enhanced_xgb_scalar","frozen_mobilenet_xgb_scalar"):
                            model.set_params(learning_rate=.05,min_child_weight=1,reg_lambda=1,reg_alpha=0,subsample=.8,colsample_bytree=.8)
                        model.fit(x[fit],y[fit],sample_weight=weights[fit])
                        model.set_params(n_jobs=1,**({"device":"cpu"} if isinstance(model,XGBRegressor) else {}))
                        scores=model.predict(x)
                    def predictor(im,model=model,extract=extract,kind=kind,cols=selected_cols):
                        xx=extract(im).reshape(1,-1)
                        if cols is not None: xx=xx[:,cols]
                        if kind in ("conditional","advantage"):
                            z=conditional_inputs(xx,len(models))
                            pred=model.predict_proba(z)[:,1] if kind=="conditional" else model.predict(z)
                            if kind=="advantage": pred[p0]=0
                            return pred
                        return model.predict(xx)[0]
                    latency_device=device if feat=="embedding" else "cpu"
                    joblib.dump({"model":model,"feature_set":feat,"selected_columns":selected_cols,
                                 "kind":kind,"models":models},out/f"{key}.joblib")
                elapsed=time.perf_counter()-start
                if not np.isfinite(scores).all(): raise ValueError(f"Nonfinite scores {key}")
                # Agreement check exercises the actual runtime feature/preprocessing path.
                if not np.allclose(predictor(images[evaluation[0]]),scores[evaluation[0]],atol=2e-4,rtol=1e-4):
                    raise ValueError(f"Runtime and batched prediction disagree: {key}")
                rows=[]; counts=[]; policies={}; selections={}
                for objective in ((trained_objective,) if scalar else active_objectives):
                    c=costs[objective]
                    if scalar:
                        policy=scalar_policy(scores[calibration],correct[calibration],c,p0,args.max_accuracy_drop)
                        choices=select_scalar(scores[evaluation],c,policy,p0)
                        choose=lambda pred,c=c,policy=policy:select_scalar(np.asarray(pred).reshape(1),c,policy,p0)
                    else:
                        policy=calibrate_utility(scores[calibration],correct[calibration],c,p0,args.max_accuracy_drop)
                        choices=select_utility(scores[evaluation],c,policy,p0)
                        choose=lambda pred,c=c,policy=policy:select_utility(np.asarray(pred).reshape(1,-1),c,policy,p0)
                    policies[objective]=policy
                    overhead=benchmark_one(images[evaluation],lambda im:choose(predictor(im)),latency_device,args.latency_samples)
                    outcome=correct[evaluation,choices]
                    reference=correct[evaluation,p0]
                    diff=outcome.astype(float)-reference.astype(float)
                    rng=np.random.default_rng(901)
                    boot=np.array([rng.choice(diff,size=len(diff),replace=True).mean()*100 for _ in range(2000)])
                    savings=100*(1-c[choices].mean()/c[p0])
                    row={"objective":objective,"router":name,"seed":seed,"baseline_model":"fp32_p0",
                        "accuracy_percent":100*outcome.mean(),"baseline_accuracy_percent":100*reference.mean(),
                        "accuracy_delta_pp":100*diff.mean(),"accuracy_delta_ci95_low":np.percentile(boot,2.5),
                        "accuracy_delta_ci95_high":np.percentile(boot,97.5),"candidate_cost":c[choices].mean(),
                        "baseline_cost":c[p0],"candidate_saving_percent":savings,
                        "cost_unit":"mJ/image" if objective=="energy" else "ms/image",
                        "router_mean_ms_pc":overhead["mean_ms"],"router_p95_ms_pc":overhead["p95_ms"],
                        "training_seconds":elapsed,"calibration_accuracy_percent":100*policy["calibration_accuracy"],
                        "meets_eval_1pp_budget":bool(outcome.mean()+1e-12>=reference.mean()-args.max_accuracy_drop),
                        "fallback_p0":policy["fallback"],"best_epoch":best_epoch,
                        "rescued_p0_errors":int((outcome & ~reference).sum()),
                        "lost_p0_correct":int((~outcome & reference).sum())}
                    rows.append(row)
                    selections[objective]=choices
                    for j,m in enumerate(models):
                        n=int((choices==j).sum())
                        counts.append({"objective":objective,"router":name,"seed":seed,"model_id":m,
                                       "selected_count":n,"selected_percent":100*n/len(evaluation)})
                    print(f"{objective:7} {name:33} s{seed} acc={row['accuracy_percent']:.2f}% delta={row['accuracy_delta_pp']:+.2f}pp saving={savings:.2f}% overhead={overhead['mean_ms']:.3f}ms",flush=True)
                result_file.write_text(json.dumps({"rows":rows,"counts":counts,"policies":policies},indent=2))
                np.savez_compressed(out/f"{key}_predictions.npz",scores=scores,evaluation=evaluation,
                                    **{f"choices_{k}":v for k,v in selections.items()})
                allrows.extend(rows); allcounts.extend(counts)
                pd.DataFrame(allrows).to_csv(out/"comparison_p0.csv",index=False)
                pd.DataFrame(allcounts).to_csv(out/"selection_counts.csv",index=False)
                del model
    df=pd.DataFrame(allrows)
    df.to_csv(out/"comparison_p0.csv",index=False)
    pd.DataFrame(allcounts).to_csv(out/"selection_counts.csv",index=False)
    summary=df.groupby(["objective","router"],sort=False).agg(
        runs=("seed","count"),accuracy_mean=("accuracy_percent","mean"),accuracy_std=("accuracy_percent","std"),
        delta_pp_mean=("accuracy_delta_pp","mean"),saving_mean=("candidate_saving_percent","mean"),
        saving_std=("candidate_saving_percent","std"),router_ms_mean=("router_mean_ms_pc","mean"),
        passing_seeds=("meets_eval_1pp_budget","sum"))
    summary.to_csv(out/"summary_p0.csv")
    print(f"Done: {out}",flush=True)


if __name__=="__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    with threadpool_limits(limits=1):
        main()
