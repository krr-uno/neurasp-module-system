# NeurASP / UNO

UNO is our PyTorch implementation of NeurASP's rule-based learning computation. It compiles an ASP program once into cached stable-model tensors, then computes observation probabilities and semantic loss in batches using PyTorch autograd on CPU or GPU. The baseline already uses vectorization and caching, but performs its semantic probability and gradient computation on CPU, even when its neural network runs on GPU.

## Review snapshot

The code selected for external review is commit **`2b1a8f96da730460bb8354fd34d8925742549753`** on `jorge/v2`. This README documents that code snapshot; subsequent development on the branch may differ. To obtain the frozen code in a separate checkout:

```bash
git clone https://github.com/krr-uno/neurasp-splitting-system.git
cd neurasp-splitting-system
git checkout --detach c851036c5d260020186265d6589ceba9b0f86f39
```

Run the commands below from the repository root.

In this repository, **UNO**, **pipeline**, and the native **`neurasp/` package** mean the same implementation. **Baseline** refers to the legacy NeurASP/MVPP implementation in `examples/baseline/`, whihc is directly obtained from the repository at: https://github.com/azreasoners/NeurASP. 

[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) provides the longer technical explanation and benchmark interpretation; its recorded results describe an earlier snapshot.

## What changed from the baseline

- **Compile before training:** the parser, compiler, and generator enumerate the full program's stable models on CPU, validate them, and encode them as input-probability indices grouped by output. Compiled models can be reused from disk across training runs.
- **Keep semantic computation on the selected device:** UNO gathers network probabilities, multiplies them within each model, and sums the model probabilities for an observation. PyTorch differentiates the mean negative log observation probability through these tensor operations into the neural network. Training forward/backward makes no Clingo calls.
- **Batch the observed-output computation:** training evaluates only models consistent with each example's observed label. Shared networks evaluate their input slots together. The baseline instead detaches neural outputs to CPU for observation-level semantic computation and sends explicit gradients back to the network's device.

This snapshot requires exactly one stable model per categorical input assignment and exactly one listed output per model. It compiles the full program; the proposed bottom/top modular splitting architecture is future work. Equal seeds do not imply identical baseline/UNO weights or training trajectories because their gradient execution and loss reductions differ.

## Environment setup

Use separate Conda environments: **Clingo 5 for the baseline** and **Clingo 6 for UNO**. Each benchmark runner launches tasks with the active Python interpreter. Use the same environment for the CPU and GPU runs of a given implementation.

If Conda has not initialized your shell, run this once, then open a new terminal:

```bash
conda init
```

Create the UNO environment from the committed [environment.yml](environment.yml):

```bash
conda env create -f environment.yml -n neurasp-gpu
conda activate neurasp-gpu
```

The YAML's default name is `neurasp2`; the `-n` override above names it `neurasp-gpu` to match the commands below. The YAML pins Clingo 6 and CUDA 13.2 PyTorch/torchvision builds. If your NVIDIA driver needs another CUDA build, adjust the wheel index and matching version suffixes as described in the YAML. These environment pins differ from the older benchmark environments recorded in `PROJECT_CONTEXT.md`.

For a fresh baseline environment, install its Clingo 5 dependencies and the Python packages used by the examples:

```bash
conda create -n neurasp5-gpu python=3.9 pip
conda activate neurasp5-gpu
python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r examples/baseline/requirements.txt numpy pandas Pillow tqdm
```

The baseline PyTorch versions above match the stack documented in [examples/baseline/README.md](examples/baseline/README.md); its requirements file allows Clingo `>=5.5,<6`. GPU runs require an NVIDIA GPU, a compatible driver, and a CUDA-enabled PyTorch build. `--device gpu` requests CUDA explicitly and fails if it is unavailable. CPU runs use `--device cpu`.

## Examples and data

The examples are launched through two benchmark runners:

- [scripts/baseline/run_benchmarks.py](scripts/baseline/run_benchmarks.py): baseline CPU/GPU runs.
- [scripts/run_pipeline_benchmarks.py](scripts/run_pipeline_benchmarks.py): UNO CPU/GPU runs and model-cache preparation.

Both runners accept the same task names, with exact capitalization:

| `--tasks` value | Learning task |
| --- | --- |
| `mnistAdd` | Infer two MNIST digits from their observed sum. |
| `member3`, `member5` | Infer three or five digits from whether a queried digit occurs among them. |
| `add2x2` | Infer four digits from the two row sums and two column sums of a 2x2 grid. |
| `card_sum_2`, `card_sum_3`, `card_sum_4` | Infer two, three, or four card identities from the sum of their rank-plus-suit values. |
| `card_prodsum_2`, `card_prodsum_3`, `card_prodsum_4` | Infer card identities from the sum of each card's rank-times-suit value. |

Digit/card identities are latent labels used for evaluation; training uses the downstream observations. Start with `mnistAdd` for a small example. UNO enumerates all latent combinations, so four-card tasks require `52^4 = 7,311,616` models and substantially more compilation time and memory.

MNIST task tuples are included; torchvision downloads MNIST images when they are missing. The examples use their configured data directories, so different tasks or implementations may download separate copies. Allow network access on first use if the images are absent.

In the current setup, card images are stored locally under `examples/card_arithmetic/images/`, and task CSVs and logic files are under `examples/card_arithmetic/tasks/`. Both implementations share this data: the baseline's `images` and `tasks` entries are relative symlinks to these directories. Preserve those symlinks when copying the repository.

The image directory must have this layout:

```text
examples/card_arithmetic/images/
├── images_train/
│   ├── playing_card_labels.csv
│   ├── <image_id>.jpg
│   └── ...
└── images_test/
    ├── playing_card_labels.csv
    ├── <image_id>.jpg
    └── ...
```

Keep the original numeric JPEG filenames: task CSVs refer to these image IDs, and each split's `playing_card_labels.csv` maps `img` IDs to card `label` values. Training and validation use `images_train/`; testing uses `images_test/`.

If the images are missing, download the input data from the [Card arithmetic dataset](https://helix.imperial.ac.uk/records/2q13m-kb120) and extract/place the images in the layout above. Images are loaded from disk; the card example does not download them automatically. For images stored elsewhere, pass `--image-dir /path/to/images` to the native card example or UNO benchmark runner (the path must contain both split directories). See the [card example README](examples/card_arithmetic/README.md) for commands.

## Run one example in all four configurations

Set `EXAMPLE` to any task name in the table. The commands use 3 epochs, batch size 16, seed `1537081416`, and one run per invocation. Each output directory must be new. Some of these paths already contain committed results for certain tasks; change the directory names before running those tasks again, or omit `--output-dir` to use a new timestamped directory automatically.

First run the baseline twice on each device:

```bash
EXAMPLE=mnistAdd
conda activate neurasp5-gpu

python scripts/baseline/run_benchmarks.py --tasks "$EXAMPLE" --device gpu --epochs 3 --batch-size 16 --seed 1537081416 --checkpoint_freq 0 --runs 1 --output-dir "benchmark_results/$EXAMPLE/baseline_gpu_run1"

python scripts/baseline/run_benchmarks.py --tasks "$EXAMPLE" --device gpu --epochs 3 --batch-size 16 --seed 1537081416 --checkpoint_freq 0 --runs 1 --output-dir "benchmark_results/$EXAMPLE/baseline_gpu_run2"

python scripts/baseline/run_benchmarks.py --tasks "$EXAMPLE" --device cpu --epochs 3 --batch-size 16 --seed 1537081416 --checkpoint_freq 0 --runs 1 --output-dir "benchmark_results/$EXAMPLE/baseline_cpu_run1"

python scripts/baseline/run_benchmarks.py --tasks "$EXAMPLE" --device cpu --epochs 3 --batch-size 16 --seed 1537081416 --checkpoint_freq 0 --runs 1 --output-dir "benchmark_results/$EXAMPLE/baseline_cpu_run2"
```

`--checkpoint_freq 0` disables card validation/checkpoint writing; final test evaluation still runs. The first and second baseline invocations keep the same seed and default task-specific cache policies. A second run can benefit from existing caches, but cache behavior differs by task; see [BASELINE_CACHING.md](examples/baseline/BASELINE_CACHING.md).

Then switch to UNO, prepare the compiled models once on CPU, and run training on CPU and GPU:

```bash
conda activate neurasp-gpu

python scripts/run_pipeline_benchmarks.py --tasks "$EXAMPLE" --device cpu --epochs 3 --batch-size 16 --runs 1 --seed 1537081416 --models-cache --compile-only --rebuild-models --output-dir "benchmark_results/$EXAMPLE/pipeline_compile_only"

python scripts/run_pipeline_benchmarks.py --tasks "$EXAMPLE" --device cpu --epochs 3 --batch-size 16 --runs 1 --seed 1537081416 --models-cache --output-dir "benchmark_results/$EXAMPLE/pipeline_cpu"

python scripts/run_pipeline_benchmarks.py --tasks "$EXAMPLE" --device gpu --epochs 3 --batch-size 16 --runs 1 --seed 1537081416 --models-cache --output-dir "benchmark_results/$EXAMPLE/pipeline_gpu"
```

`--compile-only` skips training and evaluation. `--rebuild-models` forces enumeration and rewrites the task's cache under its example directory (`.cache/*.pt`); the cache is shared by subsequent CPU and GPU runs, independently of the benchmark output directories. Keep `--rebuild-models` on the preparation command only. The training commands use `--models-cache` to load that cache; the run metrics record `models_source` as `cache` or `clingo`.

To preview commands without training or writing results, add `--dry-run` to either runner. Both also support `--help`. `--tasks` accepts multiple space-separated names; omitting it selects all ten tasks. With `--runs N`, the baseline reuses the explicit `--seed` for every repetition, while UNO increments it for each repetition (`--seed` is an alias for `--seed-base`).

## Read the results

Both runners write `runs.csv` for individual runs and `result.csv` for per-task means and population standard deviations over successful runs. Check each run's `status` and log before interpreting a measurement.

- **Baseline:** console logs are written directly in the output directory as `001_<task>.log`, with `<task>_run_metrics.jsonl` alongside the CSV reports.
- **UNO:** `manifest.json` records invocation settings and status; `logs/001_<task>.log` captures console output; `runs/001_<task>/metrics.json` contains detailed training/evaluation metrics. Aggregate reports also have JSON versions. Compile-only invocations record process timing and status without training metrics or accuracies.

`wall_seconds` in the runners' `runs.csv` measures the complete task subprocess, including imports, data preparation, cache loading, training, and evaluation. The separate UNO preparation command is outside the later training runs' timings. These commands compare different cache policies across implementations; baseline runs do not necessarily use an equivalent prebuilt cache. Equal seeds also do not guarantee identical CPU/GPU results, and the commands do not enable deterministic CUDA execution.

`latent_accuracy_pct` measures digit/card recognition. `downstream_accuracy_pct` applies the task rules to per-image argmax predictions and checks the observed result. Existing comparison reports are under [benchmark_results/comparisons/](benchmark_results/comparisons/); `PROJECT_CONTEXT.md` explains their measurement limits.

## Repository guide

| Location | Contents and role |
| --- | --- |
| `neurasp/` | UNO engine: `parser.py` translates neural declarations, `compiler.py` grounds programs, `generator.py` enumerates/validates models, `module.py` implements differentiable reasoning and network composition, `cache.py` persists compiled models, and `device.py` controls devices/precision. |
| `examples/` | Runnable UNO tasks and shared `common.py` helpers for CLI options, data loading, training, timing, and metrics. |
| `examples/mnistAdd/` | Native MNIST addition, the shared digit CNN, and addition tuple data. Its nested `baseline/` contains historical reference scripts; use `examples/baseline/` for the benchmark baseline. |
| `examples/members/` | Shared membership implementation, `member3`/`member5` entry points, and training tuples. |
| `examples/add2x2/` | Native 2x2 digit-grid task and training tuples. |
| `examples/card_arithmetic/` | Native card reasoning, CardNet, dataset loader, shared `images/` and `tasks/`, and compilation/training-step profilers. `card_direct.py` and `card_supervised.py` are additional neural-only and latent-supervised comparisons. |
| `examples/baseline/` | Legacy `neurasp.py`/`mvpp.py`, batching/device/evaluation helpers, and separate `mnistAdd/`, `member3/`, `member5/`, `add2x2/`, and `card_arithmetic/` examples. |
| `scripts/` | UNO benchmark runner and `compare_benchmarks.py` for comparing saved runs. |
| `scripts/baseline/` | Baseline benchmark runner, report exporter, and baseline runner/helper tests. |
| `scripts/latex_export/` | Exporter for four-way baseline/UNO CPU/GPU comparison tables. PDF export requires `pdflatex`. |
| `tests/` | UNO semantics, parser/compiler/generator, tensor module, cache, device, and example tests. `tests/scripts/` tests benchmark/report tools. |
| `benchmark_results/` | Committed benchmark logs, run metrics, CSV/JSON summaries, and comparison tables; also the default destination for new runs. |
| `.cache/` under each UNO example | Generated stable-model tensor caches, separate from benchmark reports. |
| `environment.yml`, `pyproject.toml` | UNO Conda dependencies and mypy configuration. Run examples from the checkout; no package build step is needed. |
| `PROJECT_CONTEXT.md`, `CLAUDE.md` | Detailed project context and architecture/development notes. Some historical paths in these files refer to earlier revisions. |

For a code review, start with `examples/mnistAdd/mnistAdd.py`, follow compilation through `neurasp/{parser,compiler,generator}.py` and execution through `neurasp/module.py`, then compare with `examples/baseline/neurasp.py` and `examples/baseline/mvpp.py`.

Run the native test suite in the UNO environment with:

```bash
conda activate neurasp-gpu
python -m unittest discover -s tests -t .
```
