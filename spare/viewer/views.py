import yaml
import numpy as np
import matplotlib.pyplot as plt

from ..galaxy import Galaxy, PhotGalaxy


__all__ = ['_ax_segmap', '_ax_rgb', '_ax_redshift', 'segmap_image', 'redshift', 'segmap_image_redshift']


def _ax_segmap(ax:plt.Axes, galaxy:Galaxy) -> None:
    galaxy_truth = np.equal(galaxy.segmap, galaxy.id)
    any_object_truth = np.not_equal(galaxy.segmap, 0)
    segmap_image = 0.5 * galaxy_truth + 0.5 * any_object_truth

    ax.imshow(segmap_image, origin='lower')

def _ax_rgb(ax:plt.Axes, galaxy:Galaxy, normalise_separate:bool, config_file:str) -> None:
    # colors
    with open(config_file) as f:
        config = yaml.safe_load(f)

    colors = ['red', 'green', 'blue']

    filters = dict()
    images = dict()
    sum = dict()
    sum_no_negative = dict()
    for color in colors:
        filters[color] = config['viewer'][color]
        images[color] = [galaxy.values[filt] for filt in filters[color]]
        sum[color] = np.sum(images[color], axis=0)
        sum_no_negative[color] = np.where(sum[color] >= 0, sum[color], 0)
    
    if normalise_separate:
        channels = {k: (image / np.max(image)) for (k, image) in sum_no_negative.items()}
    else:
        max_all = np.max([*sum_no_negative.values()])
        channels = {k: (image / max_all) for (k, image) in sum_no_negative.items()}

    rgb = np.moveaxis([channels[color] for color in colors], 0, -1)

    ax.imshow(rgb, origin='lower')

def _ax_redshift(ax:plt.Axes, galaxy:PhotGalaxy, show_text:bool) -> None:
    # get zbest and mask failures
    zbest = galaxy.zbest_reshaped()
    # use a cmap with this mask
    cmap = plt.get_cmap('Reds')
    cmap.set_bad(color='blue')
    
    ax.imshow(zbest, cmap=cmap, origin='lower')

    if show_text:
        for i, row in enumerate(zbest):
            for j, z in enumerate(row):
                _ = ax.text(j, i, f'{z:.1f}', ha='center', va='center', color='w')


def segmap_image(galaxy:Galaxy, normalise_separate:bool=True, config_file:str='config.yml') -> plt.Figure:
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    _ax_segmap(axs[0], galaxy)
    _ax_rgb(axs[1], galaxy, normalise_separate, config_file)

    fig.suptitle(f'ID: {galaxy.id}')

    fig.tight_layout()
    return fig


def redshift(galaxy:PhotGalaxy, show_text:bool=False) -> plt.Figure:
    fig, ax = plt.subplots()
    _ax_redshift(ax, galaxy, show_text)

    fig.colorbar(ax.get_images()[0])

    fig.suptitle(f'ID: {galaxy.id}')

    fig.tight_layout()
    return fig


def segmap_image_redshift(galaxy:PhotGalaxy, normalise_separate:bool=True, show_text:bool=False, config_file:str='config.yml') -> plt.Figure:
    mosaic = """
    SIR
    BBB
    """
    fig, axs = plt.subplot_mosaic(mosaic, height_ratios=[1, 0.05])

    _ax_segmap(axs['S'], galaxy)
    _ax_rgb(axs['I'], galaxy, normalise_separate, config_file)
    _ax_redshift(axs['R'], galaxy, show_text)

    fig.colorbar(axs['R'].get_images()[0], cax=axs['B'], orientation='horizontal')

    fig.tight_layout()
    fig.set_figheight(0.7 * fig.get_figheight())

    fig.suptitle(f'ID: {galaxy.id}')

    return fig
