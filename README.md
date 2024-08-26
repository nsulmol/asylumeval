# README

  This repository contains scripts for evaluating and visualizing the spectroscopic data collected by Nata. It assumes that the Asylum Research spectroscopic data has been converted to HDF5 ([ARDF-to-HDF5 Converter][https://support.asylumresearch.com/forum/asylum-research-afm/diy-programming/3003-ardf-to-hdf5-converter-for-fast-force-mapping-ffm-data]), so the used readers can function.

## Installation

1. Install poetry.
2. Install Python 3.10 (you can use pyenv to handle Python versioning).
3. Clone this repository.
4. In the repository, install via:

``` sh
poetry install
```

## Dependencies

- python
- numpy
- matplotlib
- scifireaders
- fire
- colorlog

Required, but not really necessary...
- jupyter
- ipykernel

## Usage

After installation, you should be able to visualize files via the 'visualize' method. In your poetry shell, type the following to get help info:

``` sh
visualize --help
```

(You should be able to rerun help with the options provided - e.g. 'visualize slice --help').
