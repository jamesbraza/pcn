import libgcc_fix  # isort: skip

import os
from enum import Enum
from os.path import join
from typing import Iterable, NamedTuple, Optional, Union

import lmdb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from shared.vis import plot_xyz
from tensorpack import MapData, dataflow
from tensorpack.utils.serialize import loads_msgpack

from data.shapenet import SynSet

# Feel free to change this
ABS_PATH_TO_DATASET_FOLDER: str = join(
    os.path.dirname(os.path.abspath(__file__)), "pcn"
)
NUM_GT_POINTS: int = 16_384


class DataSubset(Enum):
    """Subset of the dataset to look at."""

    TRAIN = "train"
    VALIDATION = "valid"


class PointCloudDataEntry(NamedTuple):
    """Convenient class to group data."""

    name: str
    partial: np.ndarray
    ground_truth: np.ndarray

    def discern_synset(self) -> SynSet:
        """Discern which synset the point cloud is from, returning the first match."""
        for synset in SynSet:
            if synset.value in self.name:
                return synset


class PCNShapeNetDataset:
    """Helper class to make it easy to import from PCN's ShapeNet dataset."""

    @staticmethod
    def get_lmdb_filepath(
        data_subset: DataSubset = DataSubset.TRAIN,
        path_to_dataset_dir: str = ABS_PATH_TO_DATASET_FOLDER,
    ) -> str:
        """Get the filepath."""
        return join(path_to_dataset_dir, f"{data_subset.value}.lmdb")

    @staticmethod
    def extract_from_data_buffer(data_buffer: list) -> PointCloudDataEntry:
        """Extract data from a buffer."""
        return PointCloudDataEntry(
            name=data_buffer[0], partial=data_buffer[1], ground_truth=data_buffer[2]
        )


def dataflow_interaction(lmdb_path: str) -> None:
    """Interact with the LMDB files using the tensorpack.dataflow wrappers."""
    df: MapData = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
    for data_buffer in df:
        data_entry: PointCloudDataEntry = PCNShapeNetDataset.extract_from_data_buffer(
            data_buffer
        )
        synset: SynSet = data_entry.discern_synset()
        fig1: Figure = plot_xyz(data_entry.partial)
        fig2: Figure = plot_xyz(data_entry.ground_truth)
        _ = 0  # Debug here


def direct_interaction(lmdb_path: str) -> None:
    """Directly interact with the LMDB files."""
    lmdb_env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path), readonly=True)
    with lmdb_env.begin() as lmdb_transaction:
        lmdb_cursor = lmdb_transaction.cursor()
        for _, val in lmdb_cursor:
            data_buffer: list = loads_msgpack(val)
            data_entry: PointCloudDataEntry = (
                PCNShapeNetDataset.extract_from_data_buffer(data_buffer)
            )
            synset: SynSet = data_entry.discern_synset()
            _ = 0  # Debug here


def iter_synset(
    df_iter: Union[MapData, Iterable[list]],
    synset: SynSet = SynSet.CAR,
    name_substr: Optional[str] = None,
) -> Iterable[PointCloudDataEntry]:
    """Given a dataflow, yield data entries from the specified SynSet and optionally name."""
    for data_buffer in df_iter:
        data_entry = PCNShapeNetDataset.extract_from_data_buffer(data_buffer)
        if synset.value in data_entry.name:
            if name_substr is None or name_substr in data_entry.name:
                yield data_entry


def main() -> None:
    dataflow_interaction(lmdb_path=PCNShapeNetDataset.get_lmdb_filepath())


if __name__ == "__main__":
    main()
