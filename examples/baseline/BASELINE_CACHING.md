# Stable-model caching in the baseline examples

These entry points import the repository-root `neurasp.py` and `mvpp.py`,
not the versions in `pipeline/` or `NeurASP-mnist-barebones/`.

## Default behavior

| Example | Training cache by default | Pickle path relative to working directory |
| --- | --- | --- |
| `mnistAdd/mnistAdd.py` | Disabled: explicitly passes `storeSM=False` | None read or written |
| `member3/train.py` | Enabled; deletes old disk cache before training | `saved_models/member3_stable_models.pkl` |
| `member5/train.py` | Enabled; deletes old disk cache before training | `saved_models/member5_stable_models.pkl` |
| `card_arithmetic/train.py --variant sum_2` | Enabled | `saved_models/card_sum_2_stable_models.pkl` |
| `card_arithmetic/train.py --variant sum_3` | Enabled | `saved_models/card_sum_3_stable_models.pkl` |
| `card_arithmetic/train.py --variant sum_4` | Enabled | `saved_models/card_sum_4_stable_models.pkl` |
| `card_arithmetic/train.py --variant prodsum_2` | Enabled | `saved_models/card_prodsum_2_stable_models.pkl` |
| `card_arithmetic/train.py --variant prodsum_3` | Enabled | `saved_models/card_prodsum_3_stable_models.pkl` |
| `card_arithmetic/train.py --variant prodsum_4` | Enabled | `saved_models/card_prodsum_4_stable_models.pkl` |

Deleting a pickle forces a fresh cache at startup. Models are still reused
within the first epoch and subsequent epochs and saved again. The member scripts
retain this behavior unless `--no-cache` is supplied. `mnistAdd2.py` and the
addition notebook also explicitly pass `storeSM=False`.

## Code paths and cache keys

In root `neurasp.py`:

- `NeurASP.__init__` initializes `self.stableModels = {}`.
- `learn(..., storeSM=True, ...)` loads
  `saved_models/{task}_stable_models.pkl` if present. If absent, `savePickle`
  is set and the cache is saved after each epoch.
- In the semantic-loss branch, `self.stableModels[obs[b]]` supplies cached
  models. On a miss, `dmvpp.find_k_SM_under_obs(obs[b], k=0, opt=opt)` enumerates
  all satisfying models and stores their neural-label assignment tensor.
- With `storeSM=False` and the default `method='exact'`, training calls
  `dmvpp.gradients_one_obs(...)`. In root `mvpp.py` that method calls
  `find_k_SM_under_obs(...)` every time: it creates a fresh Clingo control,
  grounds the program plus observation, enumerates models, and computes the
  gradient. Models are not retained for reuse by the next observation.
- Validation inside `learn` receives the same `storeSM` value.
- `calculate_accuracies(..., storeSM=True, ...)` has a separate default.
  Card testing calls it directly, so that call also needs `storeSM=False`.

The cache key is the **observation string**, not the image or example index.
Different images with the same downstream label can share the same models.
Network probabilities and gradients are still computed using current weights.

Addition observations are `:- not addition(i1, i2, LABEL).` in `mnistAdd.py`.
Membership observations include the queried digit, Boolean target, and
`check(DIGIT).` fact in each member `dataGen.py`. Card observations are
`:- not result(LABEL).` in `card_arithmetic/dataset.py`.

Paths depend on the working directory. In the manual commands below, member
caches are inside their example folders, and card caches are at the repository
root. The benchmark runner runs cards inside its batch output directory, so
their caches live under that directory's `saved_models/`. Later repetitions in
the same batch can reuse them. Root card pickles are not read by those
runner-launched card processes.

## Run without caching

From the repository root:

```bash
conda activate neurasp5-gpu

# All nine tasks, including all six card variants:
python scripts/baseline/run_benchmarks.py --device gpu --no-cache --runs 1

# Inspect the commands without training:
python scripts/baseline/run_benchmarks.py --device gpu --no-cache --dry-run

# Individual examples:
python examples/baseline/mnistAdd/mnistAdd.py --device gpu --no-cache
(cd examples/member3 && python train.py --device gpu --no-cache)
(cd examples/member5 && python train.py --device gpu --no-cache)
python examples/baseline/card_arithmetic/train.py --device gpu --no-cache \
    --image_dir examples/card_arithmetic/images --variant sum_2
```

For cards, choose any variant in the table with the same `--no-cache` flag.
To run seven tasks on GPU with a shared training seed, excluding both four-card
variants:

```bash
python scripts/baseline/run_benchmarks.py --device gpu --no-cache --runs 1 \
    --checkpoint_freq=2147483647 --seed=1537081416 \
    --tasks mnistAdd member3 member5 card_sum_2 card_sum_3 card_prodsum_2 card_prodsum_3
```

The runner forwards `--seed` to every task and `--checkpoint_freq` only to cards.
Membership retains its existing fixed training-subset selection (NumPy seed 1);
the supplied seed controls network initialization and PyTorch randomness.
The large checkpoint interval suppresses periodic validation but retains
NeurASP's first-batch validation and possible checkpoint save. Use
`--checkpoint_freq=0` to disable training validation/checkpoint saves entirely.
Final test evaluation still runs. Add `--dry-run` to inspect the commands.

Use `--device cpu` for CPU runs in the same environment.
Addition accepts the flag for a consistent runner interface; it already disables
caching. Other default training cache settings are preserved.

The flag passes `storeSM=False` into training and the separate card test call.
Existing pickle files are ignored and left in place. Member scripts skip
old-cache deletion in this mode. The runner records
`Stable-model caching disabled` in per-run result notes.

For Python callers, pass `storeSM=False` to both `learn` and any separate
`calculate_accuracies` call. Previously `calculate_accuracies` attempted to read
`self.stableModels` even with `storeSM=False`; the flag only prevented insertion.
It now bypasses both reads and writes, including when an object has an existing
cache from earlier calls.

MNIST downstream testing uses `benchmark_evaluation.evaluate_downstream` to
apply task rules to predicted digits and does not use the stable-model cache.

## Benchmark interpretation and verification

Uncached exact execution grounds and enumerates models for every observation,
including repeated labels and every epoch. This can substantially increase
runtime, especially for `member5` and four-card variants. Card validation also
recomputes models at each checkpoint; total process time includes this cost and
test evaluation. The flag leaves vectorized probability and gradient computation
intact. Neural `.pth` checkpoints and result files are separate from this cache.

`tests/test_baseline_cache.py` uses a small real Clingo program to verify repeated
solving, bypass of existing memory/disk caches, preserved cached behavior, and
matching weights for cached and uncached exact training. It does not establish
full-dataset runtime or accuracy.
