# Baselines on `jorge/v2`

These examples and their original NeurASP engine were copied from `ravi/generator`.
The pipeline examples remain in their existing directories. Baseline runners and
reporting tools are in `scripts/baseline/`.

From the repository root, use the baseline environment (Clingo 5, not the
pipeline's Clingo 6 environment):

```bash
conda activate neurasp5-gpu  # or neurasp5 for CPU
python scripts/baseline/run_benchmarks.py --dry-run
python scripts/baseline/run_benchmarks.py --device cpu --runs 1
python scripts/baseline/run_benchmarks.py --device gpu --tasks mnistAdd member3 member5 --epochs 1 --batch-size 16
```

Reports go to `benchmark_results/baseline_<device>_<timestamp>/` by default.
The runner uses the active Python interpreter; switch environments when running
`scripts/run_pipeline_benchmarks.py`. Baseline engine and helper imports resolve
inside this directory. The original requirements are in `requirements.txt` here;
the existing baseline environments also provide torch, torchvision, numpy,
pandas, Pillow, and tqdm.

Card `images/` and `tasks/` are relative symlinks to the existing repository data
in `examples/card_arithmetic/`; only data are shared, not example code or engines.
MNIST is downloaded by torchvision into this directory's `data/` when needed.
For individual legacy examples, run from their own directory as documented below.

---

# Examples
The inference and learning examples are stored in separate folders. A readme file is included for each example.

## Inference Examples
* [Sudoku](https://github.com/azreasoners/NeurASP/tree/master/examples/sudoku)
* [Offset Sudoku](https://github.com/azreasoners/NeurASP/tree/master/examples/offset_sudoku)
* [Toy-car](https://github.com/azreasoners/NeurASP/tree/master/examples/toycar)

## Learning Examples
* [MNIST Addition](https://github.com/azreasoners/NeurASP/tree/master/examples/mnist)
* [Shorstest Path](https://github.com/azreasoners/NeurASP/tree/master/examples/shortest_path)
* [Sudoku Solving](https://github.com/azreasoners/NeurASP/tree/master/examples/solvingSudoku_70k)
* [Top-k](https://github.com/azreasoners/NeurASP/tree/master/examples/top_k)
* [Most Reliable Path](https://github.com/azreasoners/NeurASP/tree/master/examples/most_reliable_path)
* Examples from NeuroLog paper: [add2x2](https://github.com/azreasoners/NeurASP/tree/master/examples/add2x2), [apply2x2](https://github.com/azreasoners/NeurASP/tree/master/examples/apply2x2), [member3](https://github.com/azreasoners/NeurASP/tree/master/examples/member3), [member5](https://github.com/azreasoners/NeurASP/tree/master/examples/member5)
  * To test the examples from NeuroLog paper using NeurASP, simply execute the following command in each folder.
  ```
  python train.py
  ```


## Baseline benchmark devices and downstream evaluation

Use `neurasp5` for the existing CPU environment or `neurasp5-gpu` for CUDA.
The latter preserves Python 3.9 and Clingo 5.5.2 and uses PyTorch 2.8.0+cu128,
torchvision 0.23.0+cu128, and torchaudio 2.8.0+cu128. It was created with:

```bash
conda create -n neurasp5-gpu --clone neurasp5 --offline -y
conda activate neurasp5-gpu
python -m pip install --upgrade torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

These are the [official CUDA 12.8 wheel versions](https://pytorch.org/get-started/previous-versions/#v280).
The original `neurasp5` and pipeline environments are unchanged. For CPU versus
GPU timings with a matching PyTorch build, run both devices in `neurasp5-gpu`;
comparisons with historical CPU results also include a PyTorch version change.

From the repository root:

```bash
conda activate neurasp5-gpu
python scripts/baseline/run_benchmarks.py --device gpu --tasks mnistAdd --runs 1
# Omit --tasks to run all ten tasks sequentially, with logs and CSV summaries.
python scripts/baseline/run_benchmarks.py --device gpu --runs 1

# Individual examples; member scripts retain their existing working directories.
python examples/baseline/mnistAdd/mnistAdd.py --device gpu
(cd examples/member3 && python train.py --device gpu)
(cd examples/member5 && python train.py --device gpu)
python examples/baseline/card_arithmetic/train.py --device gpu --image_dir examples/card_arithmetic/images --variant sum_2
```

All four entry points and the runner now default to CPU. Cards retain `--cpu`
as an alias; their previous automatic GPU selection is replaced by explicit
`--device gpu`. An unavailable GPU is an error, never a silent CPU fallback.
Default training batch sizes are 1 tuple for addition/members and 32 for cards.
Use runner options `--epochs` and `--batch-size` to override all selected tasks.
For example, `python scripts/baseline/run_benchmarks.py --device gpu --tasks mnistAdd member3 member5 card_sum_2 card_sum_3 --epochs 3 --batch-size 16 --seed 1537081416 --checkpoint_freq 0`
runs each task once; checkpoint frequency 0 disables card validation/checkpoints
(addition/members already disable these). Final test evaluation still runs. `benchmark_device.py` selects the device, disables
TF32 for full-float32 comparison with the pipeline, and synchronizes timers.
The neural networks run on the GPU; baseline ASP/semantic-gradient work remains
on CPU. Default training cache settings are preserved. Use `--no-cache` on these
entry points or the runner to disable stable-model reuse. See
[Baseline stable-model caching](BASELINE_CACHING.md) for the code paths, disk
locations, and uncached benchmark commands.

New runner results use `benchmark_results/baseline_<cpu|gpu>_<timestamp>/`, compatible
with `scripts/export_benchmark_latex.py` auto-selection. Logs include the actual
device and downstream evaluation metadata. The runner rejects CPU results when
GPU was requested. The existing runner CSV files contain per-run and summary
accuracies and elapsed time.

Addition and members share `benchmark_evaluation.py` with the pipeline. Addition
uses 5,000 existing test pairs; member3/member5 each use 5,000 deterministic,
balanced held-out MNIST test tuples (seed 0), as documented in `pipeline/README.md`.
These member test sets are newly defined, not verified paper test sets.
Updated runs report downstream as well as latent accuracy. Historical results
remain unchanged and need reruns to obtain previously unmeasured accuracies.

## Add2x2

The runner includes Add2x2; see [its README](add2x2/README.md) for data and
evaluation details. Select it with `--tasks add2x2`. It accepts the same device,
epoch, batch-size, seed, and cache arguments as member3/member5.
