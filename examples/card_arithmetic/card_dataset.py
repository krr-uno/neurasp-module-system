"""Load the existing card task CSVs and images for native pipeline training."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import io

TASK_DIR = Path(__file__).resolve().parent / "tasks"
ROOT = Path(__file__).resolve().parents[2]
VARIANTS = tuple(
    f"{operation}_{count}" for operation in ("sum", "prodsum") for count in (2, 3, 4)
)


def add_card_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options every card example takes: the variant, the image directory,
    weight decay and whether to preload the images.
    """
    parser.add_argument("--variant", choices=VARIANTS, default="sum_2")
    parser.add_argument(
        "--image-dir", type=Path, default=ROOT / "examples/card_arithmetic/images"
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--preload-images",
        action="store_true",
        help="decode every image file of each split once, before training, and keep "
        "them in memory (about 2-3 GB for a training split)",
    )


def card_images_to(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a batch of uint8 card images to device and scale them to [0, 1] there.

    The images cross to the device as bytes, a quarter of the size of float32, and
    are divided by 255 there. On CUDA this rounds some values differently from the
    CPU (CUDA multiplies by the reciprocal of 255), so GPU results differ slightly
    from CPU runs.
    """
    assert images.dtype == torch.uint8, f"expected uint8 images, got {images.dtype}"
    return images.to(device, non_blocking=True).float() / 255.0


class CardArithmetic(Dataset[tuple[torch.Tensor, int, int, torch.Tensor]]):
    """Card tuples of a variant's split. An item is (images, column, result, cards): the
    stacked card images, uint8 of shape (cards, 3, 274, 174), to scale with
    card_images_to; the output column of the result, its index in result_values, for
    training; the result itself, for evaluation; and each card's class 0..51.
    """

    def __init__(
        self,
        image_dir: Path,
        variant: str,
        split: str,
        result_values: list[int],
        task_dir: Path = TASK_DIR,
        preload: bool = False,
    ) -> None:
        """Load the variant's split CSV and the card labels of its image directory.

        The val split reads the training images, as the original examples do. With
        preload, every image file the split uses is decoded once, here, into one uint8
        tensor; otherwise each item reads its images from disk. Raises ValueError on an
        unknown split, a wrong number of card columns, unknown card labels, or results
        that are not in result_values.
        """
        if split not in ("train", "val", "test"):
            raise ValueError("Card split must be train, val or test")
        # Validation intentionally uses the original training image directory.
        self.image_dir = (
            Path(image_dir) / f"images_{'train' if split == 'val' else split}"
        )
        data = pd.read_csv(Path(task_dir) / f"card_{variant}_{split}.csv")
        card_columns = [column for column in data if column.startswith("card")]
        if len(card_columns) != int(variant.rsplit("_", 1)[1]):
            raise ValueError("The task CSV has an unexpected number of card columns")
        # Results are not zero-based, and prodsum has gaps: convert each to its column
        # once here, which also rejects results that are not in result_values.
        result_to_column = {value: column for column, value in enumerate(result_values)}
        columns = data["result"].map(result_to_column)
        if columns.isna().any():
            unknown = sorted(set(data["result"][columns.isna()]))
            raise ValueError(f"Unknown {split} result labels: {unknown}")

        suits = ["h", "c", "s", "d"]
        ranks = [str(rank) for rank in range(2, 11)] + ["j", "q", "k", "a"]
        semantic_to_num = {
            rank + suit: rank_index + suit_index * len(ranks)
            for suit_index, suit in enumerate(suits)
            for rank_index, rank in enumerate(ranks)
        }
        image_labels = pd.read_csv(self.image_dir / "playing_card_labels.csv")
        id_to_num = (
            image_labels.set_index("img")["label"].map(semantic_to_num).to_dict()
        )
        latent_labels = data[card_columns].apply(lambda column: column.map(id_to_num))
        if latent_labels.isna().any().any():
            raise ValueError(f"Missing or unknown card labels in {self.image_dir}")

        # Plain arrays, read per item without pandas: the image file of each card (in
        # the CSV's card order), the result column, the result and the card classes.
        self.files = data[card_columns].to_numpy(dtype=np.int64)
        self.columns = columns.to_numpy(dtype=np.int64)
        self.results = data["result"].to_numpy(dtype=np.int64)
        self.cards = torch.tensor(latent_labels.to_numpy(dtype=np.int64))
        self.preloaded: torch.Tensor | None = None
        if preload:
            self._preload()

    def _read(self, file: int) -> torch.Tensor:
        """Decode image file file.jpg of the split's directory, uint8 (3, H, W)."""
        return io.read_image(str(self.image_dir / f"{file}.jpg"))

    def _preload(self) -> None:
        """Decode every distinct image file the split uses, once, with a thread pool,
        and replace self.files by rows of the one tensor that holds them.
        """
        distinct, rows = np.unique(self.files, return_inverse=True)
        files = distinct.tolist()
        # Allocated once and filled in place: stacking a list of decoded images would
        # briefly hold every image twice.
        first = self._read(files[0])
        preloaded = torch.empty((len(files), *first.shape), dtype=torch.uint8)

        def decode(row: int) -> None:
            """Decode the row-th distinct file into its row of preloaded."""
            preloaded[row] = self._read(files[row])

        preloaded[0] = first
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            list(pool.map(decode, range(1, len(files))))
        self.preloaded = preloaded
        self.files = rows.reshape(self.files.shape)

    def __len__(self) -> int:
        """Return the number of card tuples."""
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Return tuple index as (images, column, result, cards), with its images
        preloaded or read from disk; see the class docstring.
        """
        files = self.files[index]
        if self.preloaded is not None:
            images = self.preloaded[torch.from_numpy(files)]
        else:
            images = torch.stack([self._read(int(file)) for file in files])
        return (
            images,
            int(self.columns[index]),
            int(self.results[index]),
            self.cards[index],
        )
