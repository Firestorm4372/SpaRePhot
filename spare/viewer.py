import yaml
import numpy as np
import matplotlib.pyplot as plt

from .galaxy import Galaxy, PhotGalaxy

def show_galaxy(galaxy:Galaxy, normalise_separate:bool=True, config_file:str='config.yml') -> plt.Figure:

    # segmap
    galaxy_truth = np.equal(galaxy.segmap, galaxy.id)
    any_object_truth = np.not_equal(galaxy.segmap, 0)
    segmap_image = 0.5 * galaxy_truth + 0.5 * any_object_truth

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

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs:list[plt.Axes]
    axs[0].imshow(segmap_image, origin='lower')
    axs[1].imshow(rgb, origin='lower')

    return fig


def show_redshift(galaxy:PhotGalaxy, show_text:bool=True) -> plt.Figure:
    zbest = galaxy.zbest_reshaped()

    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.imshow(zbest, cmap='Reds', origin='lower')

    if show_text:
        for i, row in enumerate(zbest):
            for j, z in enumerate(row):
                _ = ax.text(j, i, f'{z:.1f}', ha='center', va='center', color='w')

    return fig

