"""Score a fine-tuned Kev checkpoint with a quantized backbone (bitsandbytes NF4 or int8), for GPUs below the bf16 size.
Run inside the training image (bitsandbytes is installed on the fly if missing):
  docker run --gpus all --rm -v $PWD:/w -v ~/.cache/huggingface:/root/.cache/huggingface -w /w --entrypoint bash \
    ghcr.io/jtatman/kev-trainer:0.1.2 -c '/opt/kev/bin/python docker/score_quant.py code-verify-4b-v2 nf4'
Kev has no quantization option, so the backbone's from_pretrained is wrapped to add a BitsAndBytesConfig; the LoRA stays
unmerged on top (merging into 4-bit weights would round the delta away). The temperature is refit on the calibration
split under quantization. Output: runs/finetune/<run>-<quant>/ with preds_finetuned_<split>.jsonl from the quantized
model and preds_baseline_<split>.jsonl = the run's own full-precision fine-tuned predictions, so
`python eval_finetune.py <run>-<quant>` reports the quantization loss as "fine-tuned - baseline" with bootstrap CIs."""
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

RUN, QUANT = sys.argv[1], sys.argv[2]   # quant: nf4 | int8 | bf16 (no quantization: the serving precision)
SPLITS = ["development", "test_unseen_coder_all_tasks", "test_unseen_qwen3_5_9b_defiant_iq2m_all_tasks"]
QUESTION = "correct"   # the noul question id in finetune/kev records (as in docker/train_job.py)
SRC = Path("runs/finetune") / RUN
OUT = Path("runs/finetune") / f"{RUN}-{QUANT}"
DATA = Path("finetune/kev")

try:
    if QUANT != "bf16":
        import bitsandbytes  # noqa: F401
except ImportError:
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "-q", "bitsandbytes"], check=True)

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import BitsAndBytesConfig  # noqa: E402

QCONFIG = None if QUANT == "bf16" else {"nf4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_use_double_quant=True),
           "int8": BitsAndBytesConfig(load_in_8bit=True)}[QUANT]
_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained.__func__


def quantized_from_pretrained(cls, *args, **kwargs):
    kwargs.update(quantization_config=QCONFIG, device_map={"": 0}, dtype=torch.bfloat16)
    return _from_pretrained(cls, *args, **kwargs)


if QCONFIG is not None:
    transformers.AutoModelForCausalLM.from_pretrained = classmethod(quantized_from_pretrained)

from kev.checkpoint import LoadOptions  # noqa: E402
from kev.data import load_records  # noqa: E402
from kev.metrics import fit_temperature  # noqa: E402
from kev.benchmark import evaluate_records  # noqa: E402
from kev.predictors import LocalPredictor  # noqa: E402
from kev.suite import SERVING_CONTEXT  # noqa: E402


def probabilities(predictor, path, temperature):
    """p(true) per record in file order (same computation as docker/train_job.py row_probabilities), plus seconds/record."""
    rows, start = [], time.time()
    records = load_records(path)
    for rec in records:
        logits = predictor(rec)["logits"][QUESTION]
        z_true, z_false = logits["true"] / temperature, logits["false"] / temperature
        top = max(z_true, z_false)
        p_true = math.exp(z_true - top) / (math.exp(z_true - top) + math.exp(z_false - top))
        rows.append({"p_true": p_true, "label": bool(rec["questions"][QUESTION]["label"])})
    return rows, (time.time() - start) / max(len(records), 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC / "ids", OUT / "ids", dirs_exist_ok=True)
    predictor = LocalPredictor(str(SRC / "checkpoint"), "cuda", LoadOptions(temperature=1.0, dtype=torch.bfloat16, merge=False),
                               context=SERVING_CONTEXT)
    shutil.rmtree(OUT / "calibration", ignore_errors=True)
    _, cal_rows = evaluate_records(load_records(DATA / "calibration.jsonl"), predictor, OUT / "calibration")
    temperature = fit_temperature(cal_rows, aggregation="micro")
    result = {"run": RUN, "quant": QUANT, "temperature": temperature,
              "peak_gpu_gb": None, "seconds_per_record": {}}
    for split in SPLITS:
        rows, per_record = probabilities(predictor, DATA / f"{split}.jsonl", temperature)
        result["seconds_per_record"][split] = round(per_record, 3)
        with open(OUT / f"preds_finetuned_{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        if (SRC / f"preds_finetuned_{split}.jsonl").exists():   # older runs were not scored on every split
            shutil.copy(SRC / f"preds_finetuned_{split}.jsonl", OUT / f"preds_baseline_{split}.jsonl")
        print(split, len(rows), f"{per_record:.3f}s/rec", flush=True)
    result["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
