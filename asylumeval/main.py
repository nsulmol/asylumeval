"""Main scripting method to evaluate/visualize data."""

import sys
import logging
from colorlog import ColoredFormatter

import sidpy
import SciFiReaders as sr
import numpy as np
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


def compute_relationships(maps: dict[str, sidpy.sid.Dataset]
                          ) -> dict[str, sidpy.sid.Dataset]:
    """Compute Force vs. Bias and Current vs. Bias curve.

    Args:
        maps: dictionary of sidpy Datasets.

    Returns:
        dictionary of sidpy datasets, where we have replaced the original
            datasets with:
            - A force vs. bias 'spectroscopy'.
            - A current vs. bias 'spectroscopy.
    """
    defl = maps[DEFL_KEY]  # 'Force'
    bias = maps[BIAS_KEY]
    current = maps[CURR_KEY]

    dim = sidpy.sid.Dimension(bias[0][0], name=bias.name,
                              quantity=bias.quantity, units=bias.units,
                              dimension_type=sidpy.sid.DimensionType.SPECTRAL)

    force_bias = defl.copy()  # deep copy
    force_bias.set_dimension(2, dim)
    current_bias = current.copy()  # deep copy
    current_bias.set_dimension(2, dim)

    new_map = {'Force-Bias': force_bias, 'Current-Bias': current_bias}
    return new_map


def init_logging(log_level_str: str):
    color_formatter = ColoredFormatter(
        '%(asctime)s | %(name)s | '
        '%(log_color)s%(levelname)s%(reset)s:%(lineno)s | '
        '%(log_color)s%(message)s%(reset)s')

    log_level = getattr(logging, log_level_str.upper())
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)
    handler.setFormatter(color_formatter)
    logger.addHandler(handler)


def cli_spectral_slice_viz(filepath: str, spectra_key: str = BIAS_KEY,
                           log_level: str = 'INFO', **kwargs):
    """Load ARHDF5 file, visualize spectra using SpectralSliceStackVisualizer.

    This method assumes the ARHDF5 file consists of a list of spectral datasets
    which have been collected over a 2D image. It loads them, normalizes the
    'Deflection' channels, and visualizes them all in one UI.

    Keyword Args:
        figure: the figure we are drawing on.
        spectra_bin_size: size of binning of spectral 'slice', used to
            perform a mean() over the data. Default is 1.
        horizontal: whether we want to visualize the plots 'horizontally',
            or 'vertically'. Modifies rows and cols.
        fancy_grid: whether or not we use our 'fancier' Axes splitting to
            focus on the topographic data over the spectral.
        plots_per_row: how many plots we show in each row.
        scale_bar: whether or not we show x/y axis scale using a microscopy
            style scale bar (with units) in the image. If False, we show
            the individual axes with labels and units, like a standard
            plot.
        color_bar: whether or not we show a color_bar on the side for the
            topographic data.
        set_title: whether orn ot we write a title for the figure.
        **kwargs: additional keyword arguments passed to the matplotlib
            plot()/imshow() calls, based on the dataset title.

    Args:
        filepath: full path of filename to load.
        log_level: log level as string.
        spectra_key: dataset key we will use to show the spectrum (for choosing
            the spectrum index).

    """
    init_logging(log_level)

    maps = create_dset_map_from_arhdf5_file(filepath)
    viz = visualizer.SpectralSliceStackVisualizer(maps, spectra_key, **kwargs)
    plt.show(block=True)


def cli_spectral_stack_viz(filepath: str, log_level: str = 'INFO', **kwargs):
    """Load ARHDF5 file, visualize spectra sing SpectraStackVisualizer.

    This method assumes the ARHDF5 file consists of a list of spectral datasets
    which have been collected over a 2D image. It loads them, normalizes the
    'Deflection' channels, and visualizes them all in one UI.

    Keyword Args:
        horizontal: whether we want to visualize the plots 'horizontally',
            or 'vertically'. Modifies rows and cols.
        fancy_grid: whether or not we use our 'fancier' Axes splitting to
            focus on the topographic data over the spectral.
        plots_per_row: how many plots we show in each row.
        scale_bar: whether or not we show x/y axis scale using a microscopy
            style scale bar (with units) in the image. If False, we show
            the individual axes with labels and units, like a standard
            plot.
        color_bar: whether or not we show a color_bar on the side for the
            topographic data.
        set_title: whether orn ot we write a title for the figure.
        **kwargs: additional keyword arguments passed to the matplotlib
            plot()/imshow() calls, based on the dataset title.

    Args:
        filepath: full path of filename to load.
        log_level: log level as string.

    """
    init_logging(log_level)

    maps = create_dset_map_from_arhdf5_file(filepath)
    viz = visualizer.SpectraStackVisualizer(maps, **kwargs)
    plt.show(block=True)


def cli():
    """Wrapper to allow script call with pyproject."""
    fire.Fire({
        'slice': cli_spectral_slice_viz,
        'stack': cli_spectral_stack_viz})


if __name__ == '__main__':
    cli()
