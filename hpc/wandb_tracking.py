"""Optional, reduced-metadata W&B tracking; checkpoints remain local authority."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import uuid

from hpc.clean_protocol import write_json

SDK_VERSION = "0.23.0"


def public_config(plan, trial, seed, phase):
    # Explicit allowlist: never send run.json, args, environment or Slurm output.
    return {"study": plan["study"], "plan_sha256": plan["sha256"],
            "architecture": plan["architecture"], "initialization": "random",
            "image_size": plan["image_size"], "epochs": plan["epochs"],
            "batch_size": plan["batch_size"], "seed": seed, "phase": phase,
            "optimizer": trial["optimizer"], "learning_rate": trial["lr"],
            "weight_decay": trial["weight_decay"], "schedule": trial["schedule"],
            "cutmix_alpha": trial["cutmix_alpha"], "trial_id": trial["id"],
            "test_evaluated": False}


def public_metrics(row):
    output = {"epoch": row["epoch"], "learning_rate": row["lr"]}
    for role in ("train", "validation"):
        for field in ("accuracy", "loss", "seconds"):
            output[f"{role}/{field}"] = row[role][field]
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        output[f"memory/{field}"] = row[field]
    return output


class Tracker:
    def __init__(self, run=None, status_path=None):
        self.run = run
        self.status_path = status_path

    def log(self, row):
        if self.run is None:
            return
        try:
            self.run.log(public_metrics(row), step=row["epoch"])
        except Exception as error:
            # Do not print arbitrary SDK errors that could contain sensitive URLs.
            write_json(self.status_path, {"status": "logging_failed",
                                         "error_type": type(error).__name__,
                                         "training_continues_with_local_records": True})

    def summary(self, result):
        if self.run is not None:
            try:
                self.run.summary.update({key: result[key] for key in (
                    "status", "epochs_complete", "best_validation_accuracy",
                    "best_validation_loss", "training_seconds", "test_evaluated")})
            except Exception as error:
                write_json(self.status_path, {"status": "summary_failed", "error_type": type(error).__name__})


@contextmanager
def tracking_session(directory, config, mode="disabled", project=None, entity=None, sdk=None):
    if mode == "disabled":
        yield Tracker()
        return
    if mode not in ("online", "offline") or not project or (mode == "online" and not entity):
        raise ValueError("Choose online/offline and project; online also requires an entity")
    if sdk is None:
        import wandb as sdk
    if sdk.__version__ != SDK_VERSION:
        raise RuntimeError(f"Tracking tested with wandb=={SDK_VERSION}; install the pinned version")
    directory = Path(directory)
    identity_path = directory / "wandb_identity.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        if identity["project"] != project or identity["entity"] != entity:
            raise ValueError("Tracking destination changed for an existing trial")
    else:
        identity = {"id": uuid.uuid4().hex[:12], "project": project, "entity": entity}
        write_json(identity_path, identity)
    local = directory / "wandb_local"
    local.mkdir(exist_ok=True)
    settings = sdk.Settings(console="off", disable_code=True, disable_git=True,
        save_code=False, disable_job_creation=True, x_disable_meta=True,
        x_disable_stats=True, x_disable_machine_info=True, host="redacted",
        init_timeout=30, show_info=False, show_warnings=True,
        ignore_globs=("*.pt", "*.pth", "*.onnx", "*.engine", "code/*", "requirements.txt"))
    # Online resumes one logical trial. Offline uses separate execution segments:
    # caller replays local history, making every offline segment self-contained.
    run_id = identity["id"] if mode == "online" else uuid.uuid4().hex[:12]
    run = sdk.init(entity=entity, project=project, mode=mode, id=run_id,
        resume="allow" if mode == "online" else None,
        name=f"trial-{config['trial_id']:02d}-seed-{config['seed']}",
        group=config["study"], job_type=config["phase"],
        config={**config, "logical_run_id": identity["id"]},
        dir=str(local), settings=settings)
    status_path = directory / "wandb_status.json"
    write_json(status_path, {"status": "initialized", "mode": mode, "run_id": run_id})
    tracker = Tracker(run, status_path)
    failed = True
    try:
        yield tracker
        failed = False
    finally:
        try:
            run.finish(exit_code=1 if failed else 0)
        except Exception as error:
            write_json(status_path, {"status": "finish_failed", "error_type": type(error).__name__,
                                     "local_training_results_preserved": True})
