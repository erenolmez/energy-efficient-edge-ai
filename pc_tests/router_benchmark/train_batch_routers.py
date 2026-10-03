"""Train low-overhead routers that make one decision for a complete batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from torchvision.datasets import CIFAR100
from xgboost import XGBRegressor

from router_components import TinyCNN, split_positions


ROOT = Path(__file__).resolve().parent
MODELS = ["fp32_p0", "fp32_p10", "fp32_p20", "fp32_p30"]
FEATURE_NAMES = [
    "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
    "rgb_std_r", "rgb_std_g", "rgb_std_b",
    "gray_mean", "gray_std", "gradient_mean", "gradient_std",
    "edge_density", "saturation_mean", "batch_brightness_std",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="32,64,128")
    parser.add_argument("--budgets-pp", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--seeds", default="45,48,51")
    parser.add_argument("--training-shuffles", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--static-costs-energy", type=Path)
    parser.add_argument("--static-costs-latency", type=Path)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batches(positions, batch_size, shuffles=1, seed=0):
    rng = np.random.default_rng(seed)
    batches = []
    for repeat in range(shuffles):
        ordered = np.asarray(positions).copy()
        if repeat or shuffles > 1:
            rng.shuffle(ordered)
        batches.extend(
            ordered[start:start + batch_size]
            for start in range(0, len(ordered), batch_size)
        )
    return batches


def batch_feature_vector(images):
    rgb = np.asarray(images, dtype=np.float32) / np.float32(255)
    gray = rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140
    dx = gray[:, :, 1:] - gray[:, :, :-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]
    gradient = np.concatenate((np.abs(dx).ravel(), np.abs(dy).ravel()))
    saturation = rgb.max(axis=-1) - rgb.min(axis=-1)
    per_image_brightness = gray.mean(axis=(1, 2))
    return np.concatenate((
        rgb.mean(axis=(0, 1, 2)),
        rgb.std(axis=(0, 1, 2)),
        np.asarray((
            gray.mean(), gray.std(), gradient.mean(), gradient.std(),
            np.mean(gradient > 0.1), saturation.mean(),
            per_image_brightness.std(),
        )),
    )).astype(np.float32)


def make_thumbnail_grids(images, batches):
    source = torch.from_numpy(np.asarray(images).copy()).permute(0, 3, 1, 2).float() / 255
    small = F.interpolate(source, size=(8, 8), mode="bilinear", align_corners=False,
                          antialias=True)
    grids = []
    for batch in batches:
        selected = np.linspace(0, len(batch) - 1, 16).round().astype(int)
        tiles = small[np.asarray(batch)[selected]]
        grid = tiles.reshape(4, 4, 3, 8, 8).permute(2, 0, 3, 1, 4).reshape(3, 32, 32)
        grids.append(grid)
    values = torch.stack(grids)
    mean = torch.tensor([.5071, .4867, .4408])[None, :, None, None]
    std = torch.tensor([.2675, .2565, .2761])[None, :, None, None]
    return (values - mean) / std


def batch_targets(batches, correct, cost_order, p0_index):
    target = []
    batch_correct = []
    lengths = []
    for batch in batches:
        values = correct[batch]
        ordered = values[:, cost_order]
        first = np.where(ordered.any(axis=1), ordered.argmax(axis=1),
                         int(np.where(cost_order == p0_index)[0][0]))
        target.append(first.mean() / (len(cost_order) - 1))
        batch_correct.append(values.sum(axis=0))
        lengths.append(len(batch))
    return (np.asarray(target, np.float32), np.asarray(batch_correct, np.int32),
            np.asarray(lengths, np.int32))


def calibrate(scores, batch_correct, lengths, costs, p0, budget_pp):
    order = np.argsort(costs, kind="stable")
    p0_accuracy = batch_correct[:, p0].sum() / lengths.sum()
    minimum = p0_accuracy - budget_pp / 100
    best = (float(costs[p0]), -float(p0_accuracy), 0.0, True)
    for bias in np.linspace(-1, 1, 1601):
        thresholds = (np.arange(len(costs) - 1) + 0.5) / (len(costs) - 1) - bias
        choices = order[np.digitize(scores, thresholds)]
        accuracy = batch_correct[np.arange(len(choices)), choices].sum() / lengths.sum()
        mean_cost = np.average(costs[choices], weights=lengths)
        key = (float(mean_cost), -float(accuracy))
        if accuracy + 1e-12 >= minimum and key < best[:2]:
            best = (key[0], key[1], float(bias), False)
    return {
        "fallback": best[3], "bias": best[2],
        "calibration_accuracy": -best[1], "calibration_cost": best[0],
        "p0_accuracy": float(p0_accuracy), "minimum_accuracy": float(minimum),
        "accuracy_budget_pp": float(budget_pp),
    }


def select(scores, costs, policy, p0):
    if policy["fallback"]:
        return np.full(len(scores), p0, dtype=int)
    order = np.argsort(costs, kind="stable")
    thresholds = (np.arange(len(costs) - 1) + 0.5) / (len(costs) - 1) - policy["bias"]
    return order[np.digitize(scores, thresholds)]


def train_tinycnn(grids, targets, seed, epochs, device):
    seed_all(seed)
    model = TinyCNN().to(device)
    x = grids.to(device)
    y = torch.from_numpy(targets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.005)
    best_loss, best_state = float("inf"), None
    indices = np.arange(len(targets))
    cut = int(0.9 * len(indices))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train, valid = indices[:cut], indices[cut:]
    for _ in range(epochs):
        model.train()
        order = rng.permutation(train)
        for batch in np.array_split(order, max(1, int(np.ceil(len(order) / 128)))):
            prediction = model(x[batch])
            loss = F.mse_loss(prediction, y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            loss = F.mse_loss(model(x[valid]), y[valid]).item()
        if loss < best_loss:
            best_loss = loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train the batch TinyCNN")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    budgets = [float(value) for value in args.budgets_pp.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CIFAR100(root=args.data_dir, train=False, download=False)
    images = dataset.data
    labels = np.asarray(dataset.targets)
    data = pd.DataFrame({"true_label": labels})
    fit, calibration, evaluation = split_positions(data, args.split_seed)
    expected_ids = [f"test:{index:05d}" for index in range(len(images))]
    predictions = pd.read_csv(ROOT / "artifacts/predictions_all_fp32.csv")
    table = (
        predictions.assign(correct=predictions.true_label == predictions.predicted_label)
        .pivot(index="image_id", columns="model_id", values="correct")
        .reindex(index=expected_ids, columns=MODELS)
    )
    if table.isna().any().any():
        raise ValueError("Incomplete predictions for p0/p10/p20/p30")
    correct = table.to_numpy(bool)
    p0 = MODELS.index("fp32_p0")
    default_costs = {
        objective: pd.read_csv(ROOT / f"artifacts/costs_{objective}_all_fp32.csv")
        .set_index("model_id").loc[MODELS, "cost"].to_numpy(float)
        for objective in ("energy", "latency")
    }
    measured_costs = {}
    for objective, path, column in (
        ("energy", args.static_costs_energy, "vdd_in_energy_mj_per_image"),
        ("latency", args.static_costs_latency, "latency_e2e_ms_per_image"),
    ):
        if path is None:
            continue
        frame = pd.read_csv(path)
        required = {"batch_size", "model_id", column}
        if required - set(frame.columns):
            raise ValueError(f"{path} is missing {sorted(required - set(frame.columns))}")
        measured_costs[objective] = {
            int(batch_size): group.set_index("model_id").loc[MODELS, column].to_numpy(float)
            for batch_size, group in frame.groupby("batch_size")
        }

    def cost_vector(objective, batch_size):
        return measured_costs.get(objective, {}).get(batch_size, default_costs[objective])

    cost_order = np.argsort(default_costs["energy"], kind="stable")

    fit_batches, cal_batches, eval_batches = {}, {}, {}
    fit_features, cal_features, eval_features = {}, {}, {}
    fit_targets, cal_targets, eval_targets = {}, {}, {}
    for batch_size in batch_sizes:
        fit_batches[batch_size] = make_batches(
            fit, batch_size, args.training_shuffles, args.split_seed + batch_size
        )
        cal_batches[batch_size] = make_batches(calibration, batch_size)
        eval_batches[batch_size] = make_batches(evaluation, batch_size)
        fit_features[batch_size] = np.stack([
            batch_feature_vector(images[batch]) for batch in fit_batches[batch_size]
        ])
        cal_features[batch_size] = np.stack([
            batch_feature_vector(images[batch]) for batch in cal_batches[batch_size]
        ])
        eval_features[batch_size] = np.stack([
            batch_feature_vector(images[batch]) for batch in eval_batches[batch_size]
        ])
        fit_targets[batch_size] = batch_targets(
            fit_batches[batch_size], correct, cost_order, p0
        )
        cal_targets[batch_size] = batch_targets(
            cal_batches[batch_size], correct, cost_order, p0
        )
        eval_targets[batch_size] = batch_targets(
            eval_batches[batch_size], correct, cost_order, p0
        )

    x_fit = np.concatenate([fit_features[size] for size in batch_sizes])
    y_fit = np.concatenate([fit_targets[size][0] for size in batch_sizes])
    simple_models = {
        "gradient_stump": DecisionTreeRegressor(max_depth=1, min_samples_leaf=20,
                                                  random_state=42),
        "batch_tree_d3": DecisionTreeRegressor(max_depth=3, min_samples_leaf=20,
                                                 random_state=42),
        "batch_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.25, max_iter=2000, random_state=42)
        ),
        "batch_xgb_d2": XGBRegressor(
            n_estimators=120, max_depth=2, learning_rate=0.04,
            min_child_weight=20, reg_lambda=15, reg_alpha=0.2,
            subsample=0.85, colsample_bytree=0.85,
            tree_method="hist", device="cuda", n_jobs=4, random_state=42,
        ),
    }
    rows = []

    def save_method(name, model, score_function, artifact, kind):
        policies = {}
        for batch_size in batch_sizes:
            cal_scores = score_function(cal_features[batch_size], batch_size, "cal")
            eval_scores = score_function(eval_features[batch_size], batch_size, "eval")
            _, cal_correct, cal_lengths = cal_targets[batch_size]
            _, eval_correct, eval_lengths = eval_targets[batch_size]
            policies[str(batch_size)] = {}
            for objective in ("energy", "latency"):
                objective_costs = cost_vector(objective, batch_size)
                policies[str(batch_size)][objective] = {}
                for budget in budgets:
                    policy = calibrate(
                        cal_scores, cal_correct, cal_lengths, objective_costs, p0, budget
                    )
                    policies[str(batch_size)][objective][str(budget)] = policy
                    choices = select(eval_scores, objective_costs, policy, p0)
                    accuracy = eval_correct[np.arange(len(choices)), choices].sum() / eval_lengths.sum()
                    p0_accuracy = eval_correct[:, p0].sum() / eval_lengths.sum()
                    mean_cost = np.average(objective_costs[choices], weights=eval_lengths)
                    rows.append({
                        "router": name, "kind": kind, "batch_size": batch_size,
                        "objective": objective, "accuracy_budget_pp": budget,
                        "expected_accuracy_percent": 100 * accuracy,
                        "accuracy_delta_vs_p0_pp": 100 * (accuracy - p0_accuracy),
                        "expected_model_cost": mean_cost,
                        "expected_model_cost_saving_percent": 100 * (
                            objective_costs[p0] - mean_cost
                        ) / objective_costs[p0],
                        **{f"selected_batches_{model_id}": int((choices == index).sum())
                           for index, model_id in enumerate(MODELS)},
                    })
        metadata = {
            "name": name, "kind": kind, "artifact": str(artifact.name),
            "models": MODELS, "feature_names": FEATURE_NAMES,
            "batch_sizes": batch_sizes, "accuracy_budgets_pp": budgets,
            "policies": policies,
        }
        artifact.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    for name, model in simple_models.items():
        if name == "gradient_stump":
            column = FEATURE_NAMES.index("gradient_mean")
            model.fit(x_fit[:, [column]], y_fit)
            score_function = lambda values, _size, _split, m=model, c=column: m.predict(values[:, [c]])
        elif name == "batch_logistic":
            model.fit(x_fit, y_fit >= np.median(y_fit))
            score_function = lambda values, _size, _split, m=model: m.predict_proba(values)[:, 1]
        else:
            model.fit(x_fit, y_fit)
            if name == "batch_xgb_d2":
                model.set_params(device="cpu", n_jobs=1)
            score_function = lambda values, _size, _split, m=model: m.predict(values)
        artifact = args.output_dir / f"{name}.joblib"
        joblib.dump({"model": model, "kind": name, "models": MODELS,
                     "feature_names": FEATURE_NAMES}, artifact)
        save_method(name, model, score_function, artifact, "batch_features")

    combined_fit_batches = sum((fit_batches[size] for size in batch_sizes), [])
    fit_grids = make_thumbnail_grids(images, combined_fit_batches)
    combined_targets = np.concatenate([fit_targets[size][0] for size in batch_sizes])
    grid_cache = {}
    for split_name, mapping in (("cal", cal_batches), ("eval", eval_batches)):
        for batch_size in batch_sizes:
            grid_cache[(split_name, batch_size)] = make_thumbnail_grids(
                images, mapping[batch_size]
            )
    for seed in seeds:
        name = f"batch_tinycnn_s{seed}"
        model, validation_loss = train_tinycnn(
            fit_grids, combined_targets, seed, args.epochs, "cuda"
        )
        artifact = args.output_dir / f"{name}.pth"
        torch.save({
            "state_dict": model.state_dict(), "kind": "batch_tinycnn",
            "models": MODELS, "grid_tiles": 16, "tile_size": 8,
        }, artifact)

        def neural_scores(_values, batch_size, split_name, trained=model):
            values = grid_cache[(split_name, batch_size)].cuda()
            with torch.inference_mode():
                return trained(values).detach().cpu().numpy()

        save_method(name, model, neural_scores, artifact, "batch_thumbnail_grid")
        metadata_path = artifact.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        metadata["training_seed"] = seed
        metadata["validation_mse"] = validation_loss
        metadata_path.write_text(json.dumps(metadata, indent=2))

    pd.DataFrame(rows).to_csv(args.output_dir / "offline_policy_summary.csv", index=False)
    split_record = {
        "split_seed": args.split_seed,
        "fit_images": len(fit), "calibration_images": len(calibration),
        "evaluation_images": len(evaluation),
        "evaluation_sha256": hashlib.sha256(
            np.asarray(evaluation, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps({
        **split_record, "models": MODELS, "batch_sizes": batch_sizes,
        "budgets_pp": budgets, "seeds": seeds,
        "training_shuffles": args.training_shuffles, "epochs": args.epochs,
    }, indent=2))
    print(pd.DataFrame(rows).groupby(["router", "objective", "accuracy_budget_pp"])[
        ["accuracy_delta_vs_p0_pp", "expected_model_cost_saving_percent"]
    ].mean().to_string())


if __name__ == "__main__":
    main()
