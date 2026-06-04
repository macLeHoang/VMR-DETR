# VMR-DETR

PyTorch code for video moment retrieval and highlight detection with a DETR-style temporal localization model.

Given a video and a natural-language query, the model predicts relevant temporal windows and clip-level saliency scores. This repository includes training and inference code for QVHighlights-style data, Charades-STA, TVSum, optional audio features, a standalone evaluation script, and a small run-on-video demo.

## Repository Layout

| Path | Description |
| --- | --- |
| `vmr_detr/cli/` | Training and inference entrypoints. |
| `vmr_detr/config/` | Command-line options and experiment directory handling. |
| `vmr_detr/data/` | Dataset loaders for video/text and optional audio features. |
| `vmr_detr/modeling/` | VMR-DETR model, transformer, matcher, and text encoder code. |
| `vmr_detr/evaluation/` | Post-processing utilities for model predictions. |
| `vmr_detr/scripts/` | Bash scripts for common training and inference runs. |
| `standalone_eval/` | QVHighlights prediction-format evaluation utilities. |
| `run_on_video/` | Demo code for localizing moments in a raw video with CLIP features. |
| `tests/` | Unit tests for temporal matching, FDR span prediction, EMA scheduling, and related model utilities. |

## Environment

The codebase was developed around the pinned packages in `requirements.txt`, including PyTorch 1.9.0 and torchvision 0.10.0. A CUDA-capable GPU is recommended for training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate your environment and set `PYTHONPATH` with:

```powershell
$env:PYTHONPATH=".;$env:PYTHONPATH"
```

The shell scripts use Bash syntax. On Windows, run them from WSL, Git Bash, or another Bash-compatible shell.

For the raw-video demo, install the extra Python packages and make sure the `ffmpeg` binary is available on `PATH`:

```bash
pip install ffmpeg-python ftfy regex
```

## Data And Features

Training and inference consume JSONL annotation files plus pre-extracted feature files. The exact paths are provided through command-line options such as `--train_path`, `--eval_path`, `--v_feat_dirs`, `--t_feat_dir`, and `--a_feat_dir`.

Typical annotation fields are:

| Field | Description |
| --- | --- |
| `qid` | Query id used to load query/text features. |
| `query` | Natural-language query text. |
| `vid` | Video id used to load video and audio features. |
| `duration` | Video duration in seconds. |
| `relevant_windows` | Ground-truth windows as `[start, end]` timestamps in seconds. |
| `relevant_clip_ids` | Clip indices used by highlight/saliency supervision. |
| `saliency_scores` | Clip-level saliency annotations. |
| `domain`, `label` | TVSum-specific domain and dense label fields. |

Common feature layouts used by the scripts include:

```text
features/
  slowfast_features/<vid>.npz
  clip_features/<vid>.npz
  clip_text_features/qid<qid>.npz
  blip_video_features/<vid>.npz
  blip_text_features/qid<qid>.npz
  umt_pann_features/<vid>.npy
```

Video `.npz` files are expected to expose a `features` array. Query feature files are loaded by `qid`, and the default loader expects a token-level feature array such as `last_hidden_state`. See `vmr_detr/data/start_end_dataset.py` and `vmr_detr/data/start_end_dataset_audio.py` if you need to adapt feature names or formats.

Before running the provided scripts, edit the dataset and feature roots inside the script or override the arguments from the command line. Some scripts currently contain machine-specific absolute paths.

## Training

The main training entrypoint is:

```bash
PYTHONPATH=$PYTHONPATH:. python vmr_detr/cli/train.py [options]
```

For the configured Charades-STA-style experiment:

```bash
bash vmr_detr/scripts/train.sh
```

Important defaults in `vmr_detr/scripts/train.sh` include:

- dataset: `charades_sta`
- context mode: `video_tef`
- video features: SlowFast + CLIP
- optional BLIP feature support
- temporal-anchor query initialization
- FDR span loss
- temporal pyramid enabled
- EMA scheduling enabled

Outputs are written under `results_root`, usually `results/`, in an experiment-specific directory. Checkpoints are saved as variants of `model.ckpt`, including best and latest checkpoints when enabled by the training loop.

For QVHighlights-style audio training:

```bash
bash vmr_detr/scripts/train_audio.sh
```

For subtitle-style pretraining:

```bash
bash vmr_detr/scripts/pretrain.sh
```

For TVSum, the scripts sweep all TVSum domains (`BK`, `BT`, `DS`, `FM`, `GA`, `MS`, `PK`, `PR`, `VT`, `VU`) and several seeds, so review them before launching:

```bash
bash vmr_detr/scripts/tvsum/train_tvsum.sh
bash vmr_detr/scripts/tvsum/train_tvsum_audio.sh
```

## Inference

The main inference entrypoint is:

```bash
PYTHONPATH=$PYTHONPATH:. python vmr_detr/cli/inference.py --resume <checkpoint> --eval_path <jsonl> --eval_split_name <split>
```

The standard script accepts a checkpoint path and split name:

```bash
bash vmr_detr/scripts/inference.sh results/path/to/model_best.ckpt val
```

The audio variant is:

```bash
bash vmr_detr/scripts/inference_audio.sh results/path/to/model_best.ckpt val
```

Inference writes JSONL prediction files in the experiment results directory. The prediction format matches the standalone evaluation format:

```json
{
  "qid": 2579,
  "query": "A girl and her mother cooked while talking with each other on facetime.",
  "vid": "NUsG9BgSes0_210.0_360.0",
  "pred_relevant_windows": [[0, 70, 0.9986], [78, 146, 0.4138]],
  "pred_saliency_scores": [-0.2452, -0.3779, -0.4746]
}
```

## Standalone Evaluation

The `standalone_eval/` directory contains a sample QVHighlights-style evaluator and example predictions.

From the repository root:

```bash
bash standalone_eval/eval_sample.sh
```

This evaluates `standalone_eval/sample_val_preds.jsonl` and writes metrics similar to `standalone_eval/sample_val_preds_metrics_raw.json`.

See `standalone_eval/README.md` for the full prediction format and Codalab submission notes.

## Run On A Raw Video

The `run_on_video/` demo extracts CLIP features from a local video and runs a pretrained VMR-DETR checkpoint.

```bash
PYTHONPATH=$PYTHONPATH:. python run_on_video/run.py
```

The demo script expects these assets at the paths below. If you use different files or a different checkpoint location, edit the paths near the bottom of `run_on_video/run.py`.

```text
run_on_video/example/RoripwjYFp8_60.0_210.0.mp4
run_on_video/example/queries.jsonl
run_on_video/vmr_detr_ckpt/model_best.ckpt
```

See `run_on_video/README.md` for the demo-specific dependency note.

## Tests

Run the unit tests from the repository root:

```bash
python -m unittest discover tests
```

The tests cover temporal query anchors, temporal pyramid memory, task-aligned matching, FDR span prediction, GO-LSD losses, and EMA scheduling.

## License

This project is released under the MIT License. See `LICENSE` for details.
