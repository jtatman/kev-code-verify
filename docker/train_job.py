"""Kev delta fine-tune + calibration + scoring: one job for the training image, Colab VMs and local runs.

A port of the kev-finetune skill's Modal `run_train` without Modal. Stages (each resumable within a run):
  0. inputs     DATA_REPO (HF dataset, DATA_SUBDIR inside it) or files already in DATA_DIR
  1. train      python -m kev.train from the init checkpoint's own recipe, + public replay records
  2. calibrate  fit a temperature on calibration.jsonl (min NLL), score development.jsonl, write it into the checkpoint
  3. baseline   the init checkpoint on the same records, paired bootstrap (Kev's report shape: result.json)
  4. preds      per-row p(true) for development and every held-out-coder split (test_unseen*), fine-tuned and baseline
  5. publish    HF_REPO: checkpoint at the repo root, reports under run/; skipped if HF_REPO already holds run/DONE,
                so a container restarted after finishing (SaladCloud does this) does not train again
Outputs: OUT_ROOT/<RUN_NAME>/ (checkpoint/, result.json, train.log, preds_*.jsonl, DONE).

Required: RUN_NAME. Model: INIT_FROM (jaredpalmer/kev-0.8b), EPOCHS (1), REPLAY (2000), SEED (1), LR (0 = the
checkpoint's delta lr, capped at 5e-5), BATCH/ACCUM (0 = the checkpoint's), MAX_STATE (1664),
WEIGHTS_DTYPE (bf16 = bf16 frozen backbone), EVAL_DTYPE (fp32 | bf16), MAX_RECORDS (per split, for smoke tests).
  Kev-4B on a 24 GB card: WEIGHTS_DTYPE=bf16 EVAL_DTYPE=bf16 BATCH=1 ACCUM=8.
Paths: KEV_ROOT (/opt/kev-src), DATA_DIR (/workspace/data), OUT_ROOT (/workspace/out).
Hub: HF_TOKEN, DATA_REPO, DATA_SUBDIR (kev), HF_REPO, HF_PRIVATE (0).
Colab: BOOTSTRAP_VENV=/content/kevenv builds a clean uv venv first (Colab's system packages break Kev's torch 2.8).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KEV_REPO = "https://github.com/jaredpalmer/kev.git"
KEV_REF = "f2bb629d670f5b746f712fc05550a098526c836b"   # the commit the kev-finetune skill pins
KEV_ROOT = Path(os.environ.get("KEV_ROOT", "/opt/kev-src"))
SUITE = KEV_ROOT / "evals/v7/decision-v7"
DATA = Path(os.environ.get("DATA_DIR", "/workspace/data"))
MAX_DELTA_LR = 5e-5
QUESTION = "correct"
REQUIRED = ("train", "calibration", "development")


def test_splits():
    """Every held-out-coder split present (test_unseen_coder*, test_unseen_<coder>*): predicted, never trained on."""
    return sorted(p.stem for p in DATA.glob("test_unseen*.jsonl"))

NAME = os.environ["RUN_NAME"]
OUT = Path(os.environ.get("OUT_ROOT", "/workspace/out")) / NAME
CONFIG = {"init_from": os.environ.get("INIT_FROM", "jaredpalmer/kev-0.8b"), "epochs": int(os.environ.get("EPOCHS", 1)),
          "replay": int(os.environ.get("REPLAY", 2000)), "seed": int(os.environ.get("SEED", 1)),
          "lr": float(os.environ.get("LR", 0)), "batch": int(os.environ.get("BATCH", 0)),
          "accum": int(os.environ.get("ACCUM", 0)), "p_none_pair": 0.25,
          # Kev's default training context keeps only 384 state tokens, which drops ~30% of our code-verify records
          # (the longest code); raising it admits them (the packed limit grows by the same amount)
          "max_state": int(os.environ.get("MAX_STATE", 1664)),
          # memory: the released 4B trained a fp32 frozen backbone (~17 GB, sized for an 80 GB H100). WEIGHTS_DTYPE=bf16
          # halves it (Kev's own recipe for Kev-27B) so the 4B fits a 24 GB card; EVAL_DTYPE=bf16 is kev.serve's CUDA
          # default (probabilities within ~0.01 of fp32 per Kev's LoadOptions notes)
          "weights_dtype": os.environ.get("WEIGHTS_DTYPE", ""), "eval_dtype": os.environ.get("EVAL_DTYPE", "fp32"),
          "max_records": int(os.environ.get("MAX_RECORDS", 0))}
HF_REPO = os.environ.get("HF_REPO", "")


def stage(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kwargs):
    stage("$ " + " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def bootstrap_venv(venv):
    """Colab only: a clean uv venv with Kev at KEV_REF, then rerun this file inside it."""
    python = venv / "bin" / "python"
    if not (venv / ".kev_installed").exists():
        if not shutil.which("uv"):
            run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        if not KEV_ROOT.exists():
            run(["git", "clone", "--quiet", KEV_REPO, KEV_ROOT])
        run(["git", "-C", KEV_ROOT, "checkout", "--quiet", KEV_REF])
        run([uv, "venv", "--quiet", "--python", "3.13", venv])
        pip = [uv, "pip", "install", "--quiet", "--python", python]
        run(pip + [f"kev[serve] @ file://{KEV_ROOT}"])
        run(pip + ["flash-linear-attention==0.5.2", "triton>=3.7.1", "huggingface_hub>=1.0"])
        (venv / ".kev_installed").touch()
    os.execv(str(python), [str(python), __file__])


def already_published():
    if not HF_REPO:
        return False
    from huggingface_hub import HfApi
    try:
        return HfApi().file_exists(HF_REPO, "run/DONE")
    except Exception:  # noqa: BLE001 - missing repo or network: treat as not published
        return False


def fetch_inputs():
    """Training files from the Hub when DATA_REPO is set; optionally trimmed for a smoke test."""
    if os.environ.get("DATA_REPO"):
        from huggingface_hub import snapshot_download
        subdir = os.environ.get("DATA_SUBDIR", "kev")
        local = snapshot_download(os.environ["DATA_REPO"], repo_type="dataset", allow_patterns=[f"{subdir}/*"])
        DATA.mkdir(parents=True, exist_ok=True)
        for path in (Path(local) / subdir).glob("*.jsonl"):
            shutil.copy(path, DATA / path.name)
    missing = [s for s in REQUIRED if not (DATA / f"{s}.jsonl").exists()]
    if missing:
        raise SystemExit(f"missing inputs in {DATA}: {missing}")
    if CONFIG["max_records"]:
        for split in (*REQUIRED, *test_splits()):
            path = DATA / f"{split}.jsonl"
            lines = path.read_text().splitlines()[:CONFIG["max_records"]]
            path.write_text("\n".join(lines) + "\n")
        stage(f"smoke test: trimmed every split to {CONFIG['max_records']} records")


def train():
    checkpoint = OUT / "checkpoint"
    if (checkpoint / "head.pt").exists():
        stage("training already done")
        return str(checkpoint)
    from kev.checkpoint import Checkpoint
    init = Checkpoint(CONFIG["init_from"])
    meta, args = init.meta, init.meta.extra["args"]
    cfg = {"lr": min(args["lr"], MAX_DELTA_LR), **{k: args[k] for k in ("batch", "accum", "checkpointing")},
           **{k: CONFIG[k] for k in ("lr", "batch", "accum") if CONFIG[k]},
           **{k: CONFIG[k] for k in ("epochs", "seed", "replay", "p_none_pair", "max_state")}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps({"name": NAME, "init_from": CONFIG["init_from"], "init_resolved": init.path,
                                                 "base": meta.base, "config": cfg, "kev_ref": KEV_REF}, indent=1))
    cmd = [sys.executable, "-m", "kev.train", "--data", DATA / "train.jsonl", "--init_from", CONFIG["init_from"],
           "--out", checkpoint, "--device", "cuda", "--dtype", "bf16", "--base", meta.base, "--lora", meta.lora,
           "--head_dim", meta.head_dim, "--lora_targets", args["lora_targets"],
           "--option_isolation", int(meta.option_isolation), "--special_embeddings", int(meta.special_embeddings),
           "--weights_dtype", CONFIG["weights_dtype"] or meta.weights_dtype, "--epochs", cfg["epochs"], "--lr", cfg["lr"],
           "--batch", cfg["batch"], "--accum", cfg["accum"], "--checkpointing", cfg["checkpointing"],
           "--seed", cfg["seed"], "--p_none_pair", cfg["p_none_pair"], "--max_state", cfg["max_state"]]
    if meta.base_revision:
        cmd += ["--base_revision", meta.base_revision]
    if cfg["replay"]:
        cmd += ["--suite", SUITE, "--replay", cfg["replay"]]
    stage(f"training {meta.base} from {CONFIG['init_from']} with {json.dumps(cfg)}")
    with open(OUT / "train.log", "w") as log:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                cwd=KEV_ROOT)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.startswith(("ep", "saved", "device", "delta", "replay", "dropped")) or "Error" in line:
                print(line.rstrip(), flush=True)
        if proc.wait():
            raise SystemExit(f"kev.train failed ({proc.returncode}); see {OUT / 'train.log'}")
    return str(checkpoint)


def score(run_path, out, calibration, development):
    """The skill's score(): raw predictor, temperature fit on calibration, development scored at that temperature."""
    import gc
    import torch
    from kev.benchmark import evaluate_records
    from kev.checkpoint import LoadOptions
    from kev.metrics import fit_temperature
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT
    # serving limits, not the 384-token training default: long development rows are scored as a server would
    dtype = torch.bfloat16 if CONFIG["eval_dtype"] == "bf16" else None   # None = Kev's exact fp32 path
    predictor = LocalPredictor(run_path, "cuda", LoadOptions(temperature=1.0, dtype=dtype), context=SERVING_CONTEXT)
    try:
        _, cal_rows = evaluate_records(calibration, predictor, Path(out) / "calibration")
        temperature = fit_temperature(cal_rows, aggregation="micro")
        report, rows = evaluate_records(development, predictor, Path(out) / "development", temperature)
        per_row = {split: row_probabilities(predictor, DATA / f"{split}.jsonl", temperature)
                   for split in ("development", *test_splits())}
        return report, rows, temperature, per_row
    finally:
        del predictor
        gc.collect()
        torch.cuda.empty_cache()


def row_probabilities(predictor, path, temperature):
    """p(true) per record in file order, at the fitted temperature, straight from the predictor's logits."""
    import math
    from kev.data import load_records
    out = []
    for rec in load_records(path):
        logits = predictor(rec)["logits"][QUESTION]
        z_true, z_false = logits["true"] / temperature, logits["false"] / temperature
        top = max(z_true, z_false)
        p_true = math.exp(z_true - top) / (math.exp(z_true - top) + math.exp(z_false - top))
        out.append({"p_true": p_true, "label": bool(rec["questions"][QUESTION]["label"])})
    return out


def publish():
    """Checkpoint at the repo root (loadable as a Kev run), reports under run/, DONE last as the completion marker."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_REPO, private=os.environ.get("HF_PRIVATE", "0") == "1", exist_ok=True)
    api.upload_folder(repo_id=HF_REPO, folder_path=OUT / "checkpoint", commit_message=f"{NAME}: checkpoint")
    api.upload_folder(repo_id=HF_REPO, folder_path=OUT, path_in_repo="run", ignore_patterns=["checkpoint/*", "DONE"],
                      commit_message=f"{NAME}: reports")
    api.upload_file(path_or_fileobj=OUT / "DONE", path_in_repo="run/DONE", repo_id=HF_REPO,
                    commit_message=f"{NAME}: done")
    stage(f"published https://huggingface.co/{HF_REPO}")


def main():
    venv = os.environ.get("BOOTSTRAP_VENV")
    if venv and Path(sys.prefix) != Path(venv):
        stage(f"run {NAME}: {json.dumps(CONFIG)}")
        bootstrap_venv(Path(venv))
    if already_published():
        stage(f"{HF_REPO} already holds run/DONE; nothing to do")
        return
    import torch
    stage(f"run {NAME} on {torch.cuda.get_device_name(0)}, torch {torch.__version__}: {json.dumps(CONFIG)}")
    fetch_inputs()
    checkpoint = train()

    from kev.checkpoint import read_meta, write_meta
    from kev.data import load_records
    from kev.metrics import paired_bootstrap, probabilities_at_temperature
    calibration, development = load_records(DATA / "calibration.jsonl"), load_records(DATA / "development.jsonl")
    result = {"name": NAME, "config": CONFIG, "kev_ref": KEV_REF, "gpu": torch.cuda.get_device_name(0)}
    for label, run_path in (("finetuned", checkpoint), ("baseline", CONFIG["init_from"])):
        stage(f"scoring {label} ({run_path}) on {len(calibration)} calibration + {len(development)} development records")
        report, rows, temperature, per_row = score(run_path, OUT / label, calibration, development)
        result[label] = {"temperature": temperature, "raw": report["clean"], "calibrated": report["calibrated_clean"]}
        result[f"_{label}_rows"] = [{**r, "p": probabilities_at_temperature(r, temperature).tolist()} for r in rows]
        for split, rows_out in per_row.items():
            with open(OUT / f"preds_{label}_{split}.jsonl", "w") as f:
                for r in rows_out:
                    f.write(json.dumps(r) + "\n")
        if label == "finetuned":
            meta = read_meta(checkpoint)
            meta.temperature = temperature
            write_meta(checkpoint, meta)   # the checkpoint now serves calibrated probabilities by default
    finetuned_rows, baseline_rows = result.pop("_finetuned_rows"), result.pop("_baseline_rows")
    result["bootstrap"] = {m: paired_bootstrap(finetuned_rows, baseline_rows, metric=m, aggregation="micro")
                           for m in ("acc", "brier", "ece")}
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=str))
    for label in ("baseline", "finetuned"):
        m = result[label]["calibrated"]
        stage(f"{label:<10} T={result[label]['temperature']:.2f} acc {m.get('acc', float('nan')):.3f} "
              f"brier {m.get('brier', float('nan')):.3f} ece {m.get('ece', float('nan')):.3f}")
    for metric, boot in result["bootstrap"].items():
        stage(f"{metric} delta (finetuned - baseline): {boot.get(f'micro_{metric}_delta')} 95% CI {boot.get('ci95')}")
    (OUT / "DONE").touch()
    if HF_REPO:
        publish()
    stage("done")


if __name__ == "__main__":
    main()
