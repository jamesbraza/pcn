import os
from enum import Enum
from os.path import basename, expanduser, join
from typing import Iterator, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from shared.data_utils import load_h5
from shared.vis import plot_xyz

from data.shapenet import SynSet

# Feel free to change this
PATH_TO_DATASET_FOLDER: str = expanduser(
    "~/Downloads/PointCloud/ShapeNet/ShapeNet2048k/shapenet"
)


class DataSubset(Enum):
    """Subset of the data to look at."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class CloudType(Enum):
    """Which type of point cloud to get from."""

    GROUND_TRUTH = "gt"
    PARTIAL = "partial"


class C3DShapeNetDataset:
    """Helper class to make it easy to import from Completion3D's ShapeNet dataset."""

    @staticmethod
    def make_point_cloud_filepath(
        data_subset: DataSubset,
        cloud_type: CloudType,
        synset: SynSet,
        filename: str,
        path_to_dataset_dir: str = PATH_TO_DATASET_FOLDER,
    ) -> str:
        """Make a filepath to a point cloud in the dataset."""
        if filename[-3] != ".h5":
            filename += ".h5"
        return join(
            path_to_dataset_dir,
            data_subset.value,
            cloud_type.value,
            synset.value,
            filename,
        )

    @staticmethod
    def make_point_cloud_filepaths(
        data_subset: DataSubset,
        cloud_type: CloudType = CloudType.PARTIAL,
        synset: Optional[SynSet] = None,
        path_to_dataset_dir: str = PATH_TO_DATASET_FOLDER,
    ) -> Iterator[str]:
        """
        Make an iterator for point clouds in the dataset.

        Args:
            data_subset: Which subset of the data to import from.
            cloud_type: If you want a partial point cloud or a ground truth.
            synset: Optional synset prefix to use.
                If lefSt as None: pull from all synsets.
            path_to_dataset_dir: Local path to the downloaded dataset.

        Returns:
            Iterator over all the file paths to the specified point clouds.
        """
        abs_path_to_data_dir: str = join(
            path_to_dataset_dir, data_subset.value, cloud_type.value
        )
        names_list_filename: str = join(
            path_to_dataset_dir, f"{data_subset.value}.list"
        )
        with open(names_list_filename, "r", encoding="utf-8") as file:
            filenames_list: List[str] = []
            for line in file:
                if synset is None or synset.value in line:
                    filenames_list.append(
                        join(abs_path_to_data_dir, f"{line.strip()}.h5")
                    )
        yield from filenames_list

    @staticmethod
    def get_num_files(
        data_subset: DataSubset,
        cloud_type: Optional[CloudType] = None,
        synset: Optional[SynSet] = None,
        path_to_dataset_dir: str = PATH_TO_DATASET_FOLDER,
    ) -> int:
        """Get the number of files in some subset of the data."""
        dir_path: str = join(path_to_dataset_dir, data_subset.value)
        if cloud_type is not None:
            dir_path = join(dir_path, cloud_type.value)
            if synset is not None:
                dir_path = join(dir_path, synset.value)
        return sum(len(files) for _, _, files in os.walk(dir_path))


def save_figures(
    data_subset: DataSubset,
    cloud_type: CloudType = CloudType.PARTIAL,
    synset: Optional[SynSet] = None,
    num_figures: int = 3,
) -> None:
    """Save some figures from the dataset."""
    figures: List[Tuple[str, Figure]] = []
    for i, pc_filepath in enumerate(
        C3DShapeNetDataset.make_point_cloud_filepaths(
            data_subset, cloud_type, synset=synset
        )
    ):
        if not (i < num_figures):
            break
        point_cloud: np.ndarray = load_h5(pc_filepath)
        figures.append((basename(pc_filepath), plot_xyz(point_cloud)))

    for name, fig in figures:
        fig.savefig(f"figure_{name.split('.')[0]}.png")


if __name__ == "__main__":
    save_figures(DataSubset.TRAIN)
