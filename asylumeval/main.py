"""Main scripting method to evaluate/visualize data."""

import sys
import logging
from colorlog import ColoredFormatter

import sidpy
import SciFiReaders as sr
import numpy as np
from matplotlib.figure import Figure
import fire
import matplotlib.pyplot as plt

from asylumeval import visualizer


logger = logging.getLogger('asylumeval')  # why doesn't __name__ work?


# dataset.data_descriptor keys for our different channels
Z_DIST_KEY = 'ZSnsr'
DEFL_KEY = 'Defl'
CURR_KEY = 'Cur'
BIAS_KEY = 'Bias'


def create_dset_map_from_arhdf5_file(filepath: str):
    """Load and 'normalize' a desired arhdf5 file.

    By 'normalizing', we mean grabbing the 'Deflection' channel and calling
    normalize_deflection() on it.

    Args:
        filepath: full path of filename to load.

    Returns:
        dict of {CHANNEL: Dataset}, where CHANNEL is a str key from above,
            and Dataset is the associated sidpy.sid.Dataset.
    """
    reader = sr.ARhdf5Reader(filepath)
    datasets = reader.read(verbose=False)

    if datasets:
        maps = {}
        keys = [Z_DIST_KEY, DEFL_KEY, CURR_KEY, BIAS_KEY]

        for key in keys:
            # ds.data_descriptor -> name of dataset
            ds = [val for val in datasets if key in val.data_descriptor][0]
            maps[key] = ds

        maps = normalize_deflection(maps)
    return maps


def normalize_deflection(maps: dict[str, sidpy.sid.Dataset]
                         ) -> dict[str, sidpy.sid.Dataset]:
    """Normalize the deflection map by considering bias data.

    We look at the deflection values at the beginning of each spectroscopic run,
    where the bias is constant and at its initial DC value. We use this range to
    define our zero-val for the normalized deflection.

    NOTE: In this case, our normalization is really just mean-correcting....

    Args:
        maps: dictionary of sidpy Datasets.

    Returns:
        dictionary of sidpy datasets, where the deflection one has been
            normalized by the defined procedure.
    """
    indices = get_init_const_end_indices(maps[BIAS_KEY])  # NxM array
    # Computes the smallest end index, to be used for all.
    indices = (0, np.min(indices))
    maps[DEFL_KEY] = normalize_dataset(maps[DEFL_KEY], indices)
    return maps


def get_init_const_end_indices(ds: sidpy.sid.Dataset) -> np.array:
    """Determines indices where dataset stops being constant.

    This assumes the spectroscopic data provided is constant at the beginning,
    and finds the last index before it changes for each [NxM] spectroscopic
    array.

    Args:
        ds: sidpy Dataset we are analyzing. Assumes an [NxMxS] array, where
            S consists of specroscopic data.

    Returns:
        np.array of shape [NxMx1], where the last dimension contains the end
            index values before the spectroscopic data stops being constant.
    """

    grad = np.gradient(ds, axis=2)
    return (grad != 0).argmax(axis=2)  # Get first True of each in 2D arr


def normalize_dataset(ds: sidpy.sid.Dataset, indices: tuple[int]
                      ) -> sidpy.sid.Dataset:
    """Normalize dataset values by subtracting the mean of an index range.

    Args:
        ds: sidpy Dataset we are analyzing. Assumes an [NxMxS] array, where
            S consists of spectroscopic data.
        indices: tuple containing the indices of the spectroscopic axis S, to
            be used to compute our mean.

    Returns:
        sidpy.sid.Dataset with values normalized.
    """

    mean = np.mean(ds[:, :, indices[0]:indices[1]], axis=2)
    # expand to (NxMx1) for broadcasting
    mean = mean.reshape(mean.shape + (1,))
    return ds - mean


def visualize_experiment(maps: dict[str, sidpy.sid.Dataset], **kwargs
                         ) -> Figure:
    """Creates an ExperimentVisualizer and returns the associated figure."""

    dsets = list(maps.values())
    view = visualizer.ExperimentVisualizer(dsets, **kwargs)
    return view.fig


def load_and_visualize(filepath: str):
    """Load ARHDF5 file and visualize spectra.

    This method assumes the ARHDF5 file consists of a list of spectral datasets
    which have been collected over a 2D image. It loads them, normalizes the
    'Deflection' channels, and visualizes them all in one UI.

    Args:
        filepath: full path of filename to load.
    """

    maps = create_dset_map_from_arhdf5_file(filepath)
    fig = visualize_experiment(maps)
    plt.show(block=True)


def init_logging():
    color_formatter = ColoredFormatter(
        '%(asctime)s | %(name)s | '
        '%(log_color)s%(levelname)s%(reset)s:%(lineno)s | '
        '%(log_color)s%(message)s%(reset)s')

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(color_formatter)
    logger.addHandler(handler)


if __name__ == '__main__':
    #plt.ion()  # for interactive usage
    init_logging()
    fire.Fire(load_and_visualize)
