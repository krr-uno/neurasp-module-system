# Add2x2 baseline

Copied from `ravi/generator:examples/add2x2`. Uses the original baseline
engine in `examples/baseline/neurasp.py` and its local `mvpp.py`.
The network, ASP rules, 3,000-tuple subset (NumPy seed 1), learning rate,
and default three training epochs are preserved.

Run from the repository root (or use an absolute script path from any directory):

```bash
conda activate neurasp5-gpu  # use neurasp5 for the existing CPU environment
python examples/baseline/add2x2/train.py --device cpu
python examples/baseline/add2x2/train.py --device gpu
```

Requires the baseline Clingo 5 environment, torch, torchvision, numpy, and tqdm;
see `../requirements.txt` for the baseline solver dependencies. GPU mode requires
CUDA and errors if unavailable. Optional flags: `--epochs`, `--batch-size`
(default 1), `--seed`, and `--no-cache`. GPU timers synchronize through the
existing baseline device helper.

Data paths resolve relative to this file. The tracked
`../data/add2x2_train.txt` already matches the source branch and supplies the
MNIST indices and row/column sums. Torchvision downloads both MNIST splits
into `../data/MNIST/` if missing; an initial run then needs network access.
With both splits already installed, no download is needed. MNIST images
are ignored by Git.

Stable-model caches are stored in this example's `saved_models/` directory.
As in the source example, a cached file is removed before each training run
unless `--no-cache` disables caching entirely. Final evaluation reports
individual MNIST digit accuracy (`Test Acc`) and downstream Add2x2 accuracy.
Downstream evaluation uses 5,000 deterministic four-image tuples from the MNIST
test split (NumPy seed 0, no repeated image within a tuple). A tuple is correct
only when all four predicted row/column sums match. This is a newly defined
held-out protocol, not a paper-provided test set. Logs include its name, size,
and SHA-256 hash; evaluation does not change training randomness.

The shared benchmark runner includes `add2x2` by default. To select it alone:

```bash
python scripts/baseline/run_benchmarks.py --tasks add2x2 --device gpu --epochs 3 --batch-size 16 --seed 1537081416 --no-cache
```

The runner forwards device, epochs, batch size, seed, and cache settings as for
member3/member5. Its `--checkpoint_freq` option applies only to card tasks.
