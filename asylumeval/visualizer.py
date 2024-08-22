"""Experiment Visualizer class, to ease in visualization."""

import sidpy
import numpy as np
from math import ceil
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger(__name__)


PLOTS_PER_ROW = 3
SPECTRA_ALPHA = 0.3
RECT_ALPHA = 0.2


class SpectraStackVisualizer(object):
    """Interactive visualizer of spectroscopic stack.

    This was hacked up based on sidpy's SpectraImageVisualizer.

    Special kwargs:
    - 'scale_bar=True' is supposed to visualize the scale data with a microscopy-like
        scale bar (with units) in the image. I'm not sure the scale is right
        currently...
    - 'colorbar=True' will add a colorbar to the topographic graphs.

    Attributes:
        fig: the figure we are drawing on.
        image_axes: the axis ids for the image dimensions.
        energy_axes: the axes spectroscopic data for each spectroscopic
            dataset.
        dsets: the list of sidpy.sid.Datasets we are visualizing.
        dset: the first dataset, used for ease of reusing existing code.
        horizontal: whether we want to visualize the plots 'horizontally',
            or 'vertically'. Modifies rows and cols.
        x: topographic x-position of the spectra we are currently visualizing.
        y: topographic y-position of the spectra we are currently visualizing.
        bin_x: binning amount, used to downsample (I think).
        bin_y: binning amount, used to downsample (I think).
        line_scan: I'm not sure what this is, from original code.
        axes: the (flattened) list of Axes corresponding to the image and
            spectroscopic data we are visualizing.
    """

    def __init__(self, dsets: list[sidpy.sid.Dataset], figure=None,
                 horizontal=False, fancy_grid=True,
                 plots_per_row=PLOTS_PER_ROW, **kwargs):
        """Initialize our object."""
        dset_img_shape = dsets[0].shape[0:2]
        dset_ndim = dsets[0].ndim
        for dset in dsets:
            if not isinstance(dset, sidpy.Dataset):
                raise TypeError('all dsets should be a sidpy.Dataset object')
            elif dset.shape[0:2] != dset_img_shape:
                raise TypeError('all dsets must have same image shape')
            elif dset.ndim != dset_ndim:
                raise TypeError('all dsets must have same image dimensions')

        color_bar = kwargs.pop('color_bar', False)
        scale_bar = kwargs.pop('scale_bar', False)
        self.set_title = kwargs.pop('set_title', True)

        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        if figure is None:
            self.fig = plt.figure(**fig_args)
        else:
            self.fig = figure

        self.image_axes = []
        self.energy_axes = [[]]
        self.dsets = dsets
        self.dset = dsets[0]  # To modify code less

        self.horizontal = horizontal
        self.x = 0
        self.y = 0
        self.bin_x = 1
        self.bin_y = 1
        self.line_scan = False

        self._verify_and_setup_dataset()

        self.axes = self._create_axes(dsets, self.fig, horizontal,
                                      fancy_grid, plots_per_row,
                                      **fig_args)

        if self.set_title:
            self.fig.canvas.manager.set_window_title(self.dset.title)

        self.set_selection_image(scale_bar, color_bar, **kwargs)
        self._visualize_spectral_data()

        # Hookup callback for clicking on GUI (need to update spectra)
        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self._onclick)

    def _create_axes(self, dsets: list[sidpy.sid.Dataset], fig: Figure,
                     horizontal: bool, fancy_grid: bool,
                     plots_per_row: int, **fig_args) -> list[Axes]:
        """Create the axes as needed."""
        if fancy_grid:
            return self._create_axes_fancy(dsets, fig, horizontal,
                                           plots_per_row, **fig_args)
        return self._create_axes_simple(dsets, fig, horizontal,
                                        plots_per_row, **fig_args)

    @staticmethod
    def _create_axes_simple(dsets: list[sidpy.sid.Dataset], fig: Figure,
                            horizontal: bool, plots_per_row: int,
                            **fig_args) -> list[Axes]:
        """Create an nxm subplot figure for plotting purposes.

        This creates a simple nxm grid for plotting, making no real distinction
        between the image plot and the spectroscopic plots. It decides on the
        number of rows/cols based on the desired number of plots_per_row, and
        does not remove excess ones. So the visualization is a little bit crude.

        Args:
            dsets: list of sidpy.sid.Datasets which we will be using to
                plot.
            fig: the matplotlib Figure which we will create subplots on.
            horizontal: whether to display in horizontal or vertical mode.
            plots_per_row: the amount of datasets to plot for a given row
                (or column if in vertical mode).
            **fig_args: other arguments fed to the subplots() call.

        Returns:
            The list of created matplotlib Axes.
        """
        # Determine # of rows/cols and create initial subplots
        nrows = ceil((len(dsets) + 1) / plots_per_row)
        rows_cols = np.array([nrows, plots_per_row])

        if horizontal:
            rows_cols = rows_cols[::-1]  # Flip

        axes = fig.subplots(nrows=rows_cols[0], ncols=rows_cols[1],
                            **fig_args)
        return axes.flatten()

    @staticmethod
    def _create_axes_fancy(dsets: list[sidpy.sid.Dataset], fig: Figure,
                           horizontal: bool, plots_per_row: int,
                           **fig_args) -> list[Axes]:
        """Create a (more involved) nxm subplot figure for plotting purposes.

        This is a more involved subplotting routine (based off of
        _create_axes_simple), where we:
        - Set up a full row/col for the image plot;
        - Remove extra figures after we create the full grid, so the
        visualization is a bit cleaner.

        Args:
            dsets: list of sidpy.sid.Datasets which we will be using to
                plot.
            fig: the matplotlib Figure which we will create subplots on.
            horizontal: whether to display in horizontal or vertical mode.
            plots_per_row: the amount of datasets to plot for a given row
                (or column if in vertical mode).
            **fig_args: other arguments fed to the subplots() call.

        Returns:
            The list of created matplotlib Axes.
        """
        # Determine # of rows/cols and create initial subplots
        nrows = 1 + ceil(len(dsets) / plots_per_row)
        extra_figs_count = plots_per_row * (nrows - 1) - len(dsets)
        rows_cols = np.array([nrows, plots_per_row])

        if horizontal:
            rows_cols = rows_cols[::-1]  # Flip

        axes = fig.subplots(nrows=rows_cols[0], ncols=rows_cols[1],
                            **fig_args)

        img_rows_cols = np.array([nrows, 1])
        img_is = [0] * plots_per_row
        img_js = list(range(0, plots_per_row))

        if horizontal:
            img_rows_cols = img_rows_cols[::-1]  # Flip
            img_is, img_js = img_js, img_is  # Swap

        axes_to_remove = []
        # Get the rows/cols associated with the image (to be removed).
        for i, j in zip(img_is, img_js):
            axes_to_remove.append(axes[i, j])
            axes[i, j].remove()
        axes = [axis for axis in axes.flat if axis not in axes_to_remove]

        # Remove extra figs
        axes_to_remove = []
        extra_axes = axes[::-1][0:extra_figs_count]
        for ax in extra_axes:
            axes_to_remove.append(ax)
            ax.remove()
        axes = [axis for axis in axes if axis not in axes_to_remove]

        img_ax = fig.add_subplot(img_rows_cols[0], img_rows_cols[1], 1)
        axes.insert(0, img_ax)

        return axes

    def _verify_and_setup_dataset(self):
        """Confirm dataset meets expectations, set internal vars.

        Note: my linter does not like this method (cyclomatic complexity
        too high!). I should look into making this clean at some point...
        """
        dsets = self.dsets

        if len(dsets[0].shape) < 3:
            raise TypeError('datasets must have at least three dimensions')

        # We need one stack dim and two image dims as lists in dictionary
        image_axes = []
        for dim, axis in dsets[0]._axes.items():
            if (axis.dimension_type in [sidpy.DimensionType.SPATIAL,
                                        sidpy.DimensionType.RECIPROCAL]):
                image_axes.append(dim)

        if len(image_axes) == 1:
            self.line_scan = True
        elif len(image_axes) != 2:
            raise TypeError('We need two dimensions with dimension_type '
                            'SPATIAL: to plot an image')

        # Validate all datasets meet our expectations, removing those
        # which don't (with an error log).
        spectral_dims = []
        dsets_to_remove = []
        for dset in dsets:
            spectral_dim = []

            for dim, axis in dset._axes.items():
                if axis.dimension_type == sidpy.DimensionType.SPECTRAL:
                    spectral_dim.append(dim)

            if dset.variance is not None:
                if dset.variance.shape != dset.shape:
                    raise ValueError('Variance array must have the same '
                                     'dimensionality as the dataset')

            if len(dset.shape) == 3:
                if len(spectral_dim) != 1:
                    logging.error(f'dset: {dset.data_descriptor} contains '
                                  'more than 1 spectral dim. Dropping, as we '
                                  'cannot visualize it.')
                    dsets_to_remove.append(dset)
                else:
                    spectral_dims.append(spectral_dim[0])

        # Update self.dsets to remove undesired dsets
        self.dsets = [dset for dset in dsets if dset not in dsets_to_remove]

        self.image_axes = image_axes
        self.energy_axes = spectral_dims

        # Set up energy scales
        self.energy_scales = []
        for dset, energy_axis in zip(self.dsets, self.energy_axes):
            energy_scale = dset._axes[energy_axis].values
            self.energy_scales.append(energy_scale)

        size_x = self.dset.shape[self.image_axes[0]]
        size_y = self.dset.shape[self.image_axes[1]]

        # Set up extents, rectangle, etc.
        self.extent = [0, size_x, size_y, 0]  # Note: this flips the image?
        self.rectangle = [0, size_x, 0, size_y]
        self.scaleX = 1.0
        self.scaleY = 1.0
        self.analysis = []
        self.plot_legend = False
        if not self.line_scan:
            self.extent_rd = self.dset.get_extent(self.image_axes)

        logger.error(f'Image Dims: {self.image_axes}')
        logger.error(f'Energy Axes: {self.energy_axes}')
        logger.error(f'Extent: {self.extent}')
        logger.error(f'Extent RD: {self.extent_rd}')

    @staticmethod
    def _visualize_spectral_topography(axis: Axes,
                                       fig: Figure,
                                       dset: sidpy.sid.Dataset,
                                       image_axes: list[int],
                                       extent: list[int],
                                       extent_rd: list[float],
                                       energy_axis: int,
                                       scale_bar: bool,
                                       color_bar: bool,
                                       spectrum_idx: int = None,
                                       **kwargs):
        """Create 'spectral topography' image from spectral data.

        Given a spectral dataset consisting of spectral data obtained over a 2D
        topographical region, this method will create a 2D image representing
        some 'slice' of the spectroscopic data for each topographical point. If
        a specific spectrum index is provided, it will display the spectrum
        values at that index; otherwise, it will perform a mean along the
        energy axis.

        Args:
            axis: matplotlib Axes where you will draw the image.
            dset: sidpy dataset from which we extract this data.
            image_axes: axis ids for the image data (topographic).
            extent: 4-val int list indicating extents of the topographic
                indices. Used for plotting purposes.
            extent_rd: 4-val float list indicating extents of the topographic
                data, in real units. Used for plotting purposes.
            energy_axis: the axes of the spectroscopic data.
            spectrum_idx: spectrum image from which we create the 2D image.
            scale_bar: whether or not to replace axes dimensions with a
                single scale bar, drawn directly on the data (microscopy
                standard).
            color_bar: whether or not to show a colorbar on the side of
                the image, indicating the range of the z-dim data.
            **kwargs: additional args to feed to the matplotlib plot method.
        """
        # Create clickable image from first spectrum's mean image.
        if spectrum_idx is None:
            image = dset.mean(axis=(energy_axis))
        else:
            # Get 2D image at spectrum slice! Using take() to grab a
            # slice along the energy axis, and then squeeze to squash
            # that dimension (that has size 1).
            image = np.squeeze(np.take(dset, indices=[spectrum_idx],
                                       axis=energy_axis))

        ax_img = axis.imshow(image.T, extent=extent, **kwargs)

        # Handle colorbar
        if create_color_bar:
            create_color_bar(axis, fig, ax_img, dset)
        if create_scale_bar:
            create_scale_bar(axis, extent_rd, image_axes, dset)

        # If our 'image' is 1D instead of 2D, handle (not sure this will get
        # hit).
        if 1 in dset.shape:
            axis.set_aspect('auto')
            axis.get_yaxis().set_visible(False)
        else:
            axis.set_aspect('equal')

        # Handle x/y ticks, labels.
        axis.set_xticks(np.linspace(extent[0], extent[1], 5))
        axis.set_xticklabels(np.round(np.linspace(
            extent[0], extent[1], 5), 2))

        axis.set_yticks(np.linspace(extent[2], extent[3], 5))
        axis.set_yticklabels(np.round(np.linspace(
            extent[2], extent[3], 5), 1))

        axis.set_xlabel('{} [{}]'.format(
            dset._axes[image_axes[0]].quantity, 'px'))
        axis.set_ylabel('{} [{}]'.format(
            dset._axes[image_axes[1]].quantity, 'px'))

    def set_selection_image(self, scale_bar: bool,
                            color_bar: bool, **kwargs):
        """Create image we will use to select data to display.

        Args:
            scale_bar: whether or not to replace axes dimensions with a
                single scale bar, drawn directly on the data (microscopy
                standard).
            color_bar: whether or not to show a colorbar on the side of
                the image, indicating the range of the z-dim data.
        """
        self._visualize_spectral_topography(self.axes[0],
                                            self.fig,
                                            self.dsets[0],
                                            self.image_axes,
                                            self.extent,
                                            self.extent_rd,
                                            self.energy_axes[0],
                                            create_scale_bar,
                                            create_color_bar,
                                            spectrum_idx=None, **kwargs)

        # Place a rectangle indicating the bin we are on.
        self.rect = patches.Rectangle((0, 0), self.bin_x, self.bin_y,
                                      linewidth=1, edgecolor='r',
                                      facecolor='red', alpha=RECT_ALPHA)
        self.axes[0].add_patch(self.rect)

    def _visualize_spectral_data(self):
        """Visualize all spectra.

        Iterate through datasets and UI 'axes' (skipping first axis, as it
        is the base image).
        """
        for dset, axis, energy_axis, energy_scale in zip(
                self.dsets, self.axes[1:(1+len(self.dsets))],
                self.energy_axes, self.energy_scales):
            self._visualize_spectrum(axis, dset, energy_axis, energy_scale)
        # self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _visualize_spectrum(self, axis: Axes, dset: sidpy.sid.Dataset,
                            energy_axis: int, energy_scale: np.array):
        """Set or update visualization of spectrum on provided axis.

        Args:
            axis: matplotlib Axes object, containing all the data of a particular
                subplot.
            dset: sidpy Dataset with spectral data of interest
            energy_axis: axis of dset containing energy data.
            energy_scale: np.array containing the TODO: FINISH ME.
        """
        # Clear axis before re-drawing
        axis.clear()
        spectrum, variance = self.get_spectrum_variance(dset)

        # Transpose if necessary
        if len(energy_scale) != spectrum.shape[0]:
            spectrum = spectrum.T

        axis.plot(energy_scale, spectrum.compute())

        # add variance shadow graph
        if variance is not None:
            #3d - many curves
            if len(variance.shape) > 1:
                for i in range(len(variance)):
                    axis.fill_between(energy_scale,
                                      spectrum.compute().T[i] - variance[i],
                                      spectrum.compute().T[i] + variance[i],
                                      alpha=SPECTRA_ALPHA, **self.kwargs)
            # 2d - one curve at each point
            else:
                axis.fill_between(energy_scale,
                                  spectrum.compute() - variance,
                                  spectrum.compute() + variance,
                                  alpha=SPECTRA_ALPHA, **self.kwargs)

        axis.set_title('spectrum {}, {}'.format(self.x, self.y))

        axis.set_xlim(axis.get_xlim())
        axis.set_ylim(axis.get_ylim())
        axis.set_xlabel(dset.labels[energy_axis])  # + x_suffix)
        axis.set_ylabel(dset.data_descriptor)
        axis.ticklabel_format(style='sci', scilimits=(-2, 3))

    def get_spectrum_variance(self, dset: sidpy.sid.Dataset
                              ) -> (np.array, np.array):
        """Extract the spectrum and variance of a provided SPECTRAL dataset.

        Args:
            dset: sidpy dataset from which we extract this data.

        Returns:
            Tuple of (spectrum, variance).
        """
        if self.x > dset.shape[self.image_axes[0]] - self.bin_x:
            self.x = dset.shape[self.image_axes[0]] - self.bin_x
        if self.y > dset.shape[self.image_axes[1]] - self.bin_y:
            self.y = dset.shape[self.image_axes[1]] - self.bin_y
        selection = []

        for dim, axis in dset._axes.items():
            if axis.dimension_type == sidpy.DimensionType.SPATIAL:
                if dim == self.image_axes[0]:
                    selection.append(slice(self.x, self.x + self.bin_x))
                else:
                    selection.append(slice(self.y, self.y + self.bin_y))

            elif axis.dimension_type == sidpy.DimensionType.SPECTRAL:
                selection.append(slice(None))
            elif axis.dimension_type == sidpy.DimensionType.CHANNEL:
                selection.append(slice(None))
            else:
                selection.append(slice(0, 1))

        spectrum = dset[tuple(selection)].mean(axis=tuple(self.image_axes))

        if dset.variance is not None:
            variance = dset.variance[tuple(selection)].mean(axis=tuple(self.image_axes))
        else:
            variance = None

        return (spectrum.squeeze(), variance)

    def _onclick(self, event):
        """Update visualizations if the selected image bin has changed."""
        self.event = event

        # TODO: Remove when comfortable...
        if ax := event.inaxes:
            logger.debug(f'idx: {ax.figure.axes.index(ax)}, bounds: {ax._position.bounds}')
            logger.debug(f'xdata: {event.xdata}, ydata: {event.ydata}')

        if event.inaxes == self.axes[0]:
            x = int(event.xdata)
            y = int(event.ydata)

            x = int(x - self.rectangle[0])
            y = int(y - self.rectangle[2])

            if x >= 0 and y >= 0:
                if x <= self.rectangle[1] and y <= self.rectangle[3]:
                    self.x = int(x / (self.rect.get_width() / self.bin_x))
                    self.y = int(y / (self.rect.get_height() / self.bin_y))

                    if self.x + self.bin_x > self.dset.shape[self.image_axes[0]]:
                        self.x = self.dset.shape[self.image_axes[0]] - self.bin_x
                    if self.y + self.bin_y > self.dset.shape[self.image_axes[1]]:
                        self.y = self.dset.shape[self.image_axes[1]] - self.bin_y

                    self.rect.set_xy([self.x * self.rect.get_width() / self.bin_x + self.rectangle[0],
                                      self.y * self.rect.get_height() / self.bin_y + self.rectangle[2]])
            self._visualize_spectral_data()

    def set_legend(self, set_legend):
        self.plot_legend = set_legend

    def get_xy(self):
        return [self.x, self.y]

    def set_bin(self, bin_xy):
        old_bin_x = self.bin_x
        old_bin_y = self.bin_y
        if isinstance(bin_xy, list):
            self.bin_x = int(bin_xy[0])
            self.bin_y = int(bin_xy[1])
        else:
            self.bin_x = int(bin_xy)
            self.bin_y = int(bin_xy)

        if self.bin_x > self.dset.shape[self.image_axes[0]]:
            self.bin_x = self.dset.shape[self.image_axes[0]]
        if self.bin_y > self.dset.shape[self.image_axes[1]]:
            self.bin_y = self.dset.shape[self.image_axes[1]]

        self.rect.set_width(self.rect.get_width() * self.bin_x / old_bin_x)
        self.rect.set_height((self.rect.get_height() * self.bin_y / old_bin_y))
        if self.x + self.bin_x > self.dset.shape[self.image_axes[0]]:
            self.x = self.dset.shape[0] - self.bin_x
        if self.y + self.bin_y > self.dset.shape[self.image_axes[1]]:
            self.y = self.dset.shape[1] - self.bin_y

        self.rect.set_xy([self.x * self.rect.get_width() / self.bin_x + self.rectangle[0],
                          self.y * self.rect.get_height() / self.bin_y + self.rectangle[2]])
        self._visualize_spectral_data()

    @staticmethod
    def _closest_point(array_coord, point):
        diff = array_coord - point
        return np.argmin(diff[:,0]**2 + diff[:,1]**2)


def create_color_bar(axis: Axes, fig: Figure,
                     ax_img: AxesImage, dset: sidpy.sid.Dataset):
    """Add a colorbar to a created AxesImage, with data from dset."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(ax_img, cax=cax)
    cbar.set_label(dset.data_descriptor)


def create_scale_bar(axis: Axes, extent_rd: list[float],
                     image_axes: list[int], dset: sidpy.sid.Dataset,):
    """Replace axes data with a scale bar drawn in the image."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    axis.axis('off')

    # remove previous scale_bars
    for artist in axis.artists:
        if isinstance(artist, AnchoredSizeBar):
            artist.remove()

    extent = extent_rd
    _units = dset._axes[image_axes[0]].units

    size_of_bar_real = (extent[1] - extent[0]) / 5
    if size_of_bar_real < 1:
        size_of_bar_real = round(size_of_bar_real, 1)
    else:
        size_of_bar_real = int(round(size_of_bar_real))
    px_size = axis.get_xlim()
    size_of_bar = int((px_size[1] - px_size[0]) * (size_of_bar_real /
                                                   (extent[1] - extent[0])))

    if size_of_bar < 1:
        size_of_bar = 1
    scalebar = AnchoredSizeBar(axis.transData,
                               size_of_bar, '{} {}'.format(size_of_bar_real,
                                                           _units),
                               'lower left',
                               pad=1,
                               color='white',
                               frameon=False,
                               size_vertical=size_of_bar / 7)

    axis.add_artist(scalebar)
