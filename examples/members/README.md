# MNIST membership

Given three or five MNIST images, a query digit, and a binary membership label,
learn digit classification using ASP membership reasoning. Digit labels are
used only for evaluation.

`member.py` implements both tasks. `member3.py` and `member5.py` are small callers
that select the image count. The original `ravi/generator` branch kept separate
`member3/` and `member5/` directories, each with largely duplicated `train.py`,
`dataGen.py`, and `network.py` implementations.

Run from the repository root:

```bash
python -m examples.members.member3
python -m examples.members.member5 --device gpu
python -m examples.members.member --count 3
python -m examples.members.member --count 5 --compile-only
```

Direct script execution also works, for example:
`python examples/members/member5.py --compile-only`.
Defaults: CPU, batch 16, 3 epochs, Adam learning rate 0.001.

`member3_train.txt` and `member5_train.txt` contain the original branch's tuples:
image indices, query digit, and label (0 absent, 1 present). Training uses the
original fixed 3000-row subset (NumPy seed 1), without shuffling. MNIST images
are shared under `examples/data/MNIST` and downloaded if needed.

Both tasks reuse the MNIST addition network and image helpers.
`examples/common.py` supplies CLI options, seeding, loaders, timing, the epoch
loop, compile-cache options, and metrics output.

ASP is compiled once into a `NeurASPModule`; PyTorch sums stable-model
probabilities and backpropagates semantic loss during training. The query is
normalized to zero by rotating digit probability columns, preserving membership
while requiring only 10^3 or 10^5 stable models instead of ten copies per query.
The compiled input has three or five digit slots; output columns are absent and
present. Both tasks cache their models under `.cache/`, using distinct cache
entries named `member3` and `member5`.

Metrics use the existing examples' format and output options. Latent accuracy
measures digit classification on the MNIST test split and first 1000 training
images. Downstream accuracy uses the source branch's 5000 fixed, balanced
membership queries over test images (seed 0), applying membership to argmax
digit predictions. These queries are generated, not paper-provided tuples.
