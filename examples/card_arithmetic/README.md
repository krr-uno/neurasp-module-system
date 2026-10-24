# Card arithmetic
The Card arithmetic problem is a collection of challenging tasks that use 174x274 pixel playing card images as inputs. In each of them, the downstream result represents an arithmetic operation involving the rank and suit values of the input cards.

For each task, there are 2-, 3- and 4-card variations, and the difficulty increases accordingly. The rank values are 2-10 for the corresponding numerical ranks, and 11-14 for the Jack, Queen, King and Ace. The suit values vary depending on the task. The end result is the sum of all the card values.

## Card sum
The suits Diamonds, Clubs, Spades and Hearts are given values 0, 13, 26, 29. The card value is the _sum_ of its rank and suit value. Thus, each card is given a unique value.

## Card prodsum
The suits Diamonds, Clubs, Spades and Hearts are given values 1, 2, 3, 4. The card value is the _product_ of its rank and suit value. Thus, two different cards could be assigned the same value. E.g. the 10 of Diamonds ($10\times1=10$) has the same value as the 5 of Spades ($5\times2$). This dataset is therefore well suited for investigating reasoning shortcuts, where the neural network might learn incorrect latent labels that lead to the correct downstream label.

## Images and task data

The current setup stores images in `examples/card_arithmetic/images/` and task CSVs and logic files in `examples/card_arithmetic/tasks/`. The baseline shares these folders through the relative symlinks `examples/baseline/card_arithmetic/images` and `examples/baseline/card_arithmetic/tasks`.

The default image directory contains:

```text
images/
├── images_train/
│   ├── playing_card_labels.csv
│   ├── <image_id>.jpg
│   └── ...
└── images_test/
    ├── playing_card_labels.csv
    ├── <image_id>.jpg
    └── ...
```

Keep the original numeric JPEG filenames. The task CSVs refer to image IDs, and `playing_card_labels.csv` in each split maps `img` IDs to card `label` values. Training and validation both read `images_train/`; testing reads `images_test/`. There is no separate validation image folder.

If the data are missing, download the input files from the [Card arithmetic dataset](https://helix.imperial.ac.uk/records/2q13m-kb120), place the task files in `tasks/`, and arrange the images as shown above. Images are loaded from disk and are not downloaded automatically.

## How to run

From the repository root, use the images in the default location:

```bash
python -m examples.card_arithmetic.card_arithmetic --variant sum_2
```

For images stored elsewhere, pass the parent directory containing `images_train/` and `images_test/`:

```bash
python -m examples.card_arithmetic.card_arithmetic --variant sum_2 --image-dir /path/to/images
```

Variants are `sum_2`, `sum_3`, `sum_4`, `prodsum_2`, `prodsum_3`, and `prodsum_4`. The default device is CPU; use `--device gpu` for GPU training. Run with `--help` for all options.
