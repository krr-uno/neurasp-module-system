# MNIST Addition
The digit addition problem `mnistAdd` is a simple but illustrative example used in [(Manhaeve et al. 2018)](https://arxiv.org/abs/1805.10872) to illustrate DeepProbLog’s ability to do both logical reasoning and deep learning. The task is, given a pair of digit images (MNIST) and their sum as the label, let a neural network learn the digit classification of the input images.

This problem can be extended to `mnistAdd2` where each number consists of 2 digit images.

## File Description
* data: training and testing tuples of MNIST indices followed by the label: `train_data.txt` and `test_data.txt` (mnistAdd, `idx1 idx2 sum`). The MNIST images are downloaded to `data/MNIST` on first use (gitignored).
* mnistAdd.py: the `mnistAdd` example on this repo's pipeline: `parse_neurasp_program` -> Clingo grounding -> `ModuleGenerator` -> `NeurASPProgram([(network, 2)], module)` -> PyTorch training.
* network.py: the digit classifier `Net` and `testNN`, which measures digit accuracy.
* dataGen.py, mnist.ipynb, baseline/: the original NeurASP versions, kept for reference. They use the original `NeurASP` class, which is not in this repo, and do not run here.

## How to run
From the repository root:
```
python -m examples.mnistAdd.mnistAdd
```
Options: `--device cpu|gpu` (default cpu), `--batch-size` (16), `--epochs` (1), `--lr` (0.001), `--seed` (1), `--output-dir` (default `benchmark_results/pipeline_<task>_<device>_batch<N>_<timestamp>/`, holding `metrics.json` and `runs.csv`).

Latent accuracy is digit classification on the MNIST test split (and on the first 1000 training images). For mnistAdd, downstream accuracy applies addition to the argmax digits of the held-out `test_data.txt` pairs.
