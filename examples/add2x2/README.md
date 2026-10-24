# MNIST Add2x2

Learn digit classification from four MNIST images arranged as a 2x2 grid and
observations of their two row sums and two column sums. Digit labels are used
only for evaluation. This is the pipeline counterpart of `examples/baseline/add2x2`.

Run in the pipeline Clingo 6 environment, from the repository root:

```bash
python -m examples.add2x2.train
python -m examples.add2x2.train --device gpu --epochs 3 --batch-size 16
python -m examples.add2x2.train --compile-only
```

Direct execution also works from any directory:
`python /path/to/repo/examples/add2x2/train.py`.
Defaults match the pipeline membership examples: CPU, batch 16, three epochs,
and Adam learning rate 0.001. The network and image helpers are shared with
MNIST addition. `examples/common.py` supplies training arguments, seeding,
loaders, timers, the epoch loop, and metrics files.

`add2x2_train.txt` is the baseline's original training data: four MNIST indices,
then row 1, row 2, column 1, and column 2 sums. Training uses the same 3,000-row
subset (NumPy seed 1), without shuffling. MNIST images are shared under
`examples/data/MNIST/` and downloaded by torchvision if missing.

The ASP program uses the pipeline's digit slots 1 through 4. Compilation groups
all 10,000 digit assignments by their complete four-sum observation. Output
columns are the sorted reachable sum combinations. During training,
`NeurASPModule.forward_single_output` sums the model probabilities for the
observed combination, and semantic loss backpropagates through the shared digit
network. Compilation is cached in `.cache/add2x2.pt`.

All shared options are supported, including `--seed`, `--deterministic`,
`--learning-rate`, `--precision`, `--workers`, enumeration options,
`--models-cache`/`--no-models-cache`, `--rebuild-models`, `--compile-only`, and
`--output-dir`. Compile-only runs need no MNIST data and write no training metrics.

Metrics follow the pipeline format (`metrics.json` and `runs.csv`). Latent
accuracy is digit classification on the MNIST test split and first 1,000 training
images. Downstream accuracy uses exactly the baseline's 5,000 fixed held-out
four-image tuples (NumPy seed 0, no repeated image within a tuple). All four
predicted row/column sums must match. This generated protocol is not a
paper-provided test set; its name, size, and SHA-256 are recorded for comparison.

The pipeline benchmark runner includes Add2x2 by default when `--tasks` is
omitted. To select it alone:

```bash
python scripts/run_pipeline_benchmarks.py --tasks add2x2 --device gpu --epochs 3 --batch-size 16
```

The runner forwards the shared training, device, precision, enumeration and cache
options, and collects Add2x2 metrics in the same way as the other pipeline tasks.
