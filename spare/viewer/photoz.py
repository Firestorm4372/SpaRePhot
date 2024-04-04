import numpy as np
import matplotlib.pyplot as plt

from ..galaxy import PhotGalaxy
from . import views


__all__ = ['_ax_single_chi2',
           '_total_chi2_values',
           'pixel_chi2', 'total_chi2', 'views_and_total_chi2']


def _ax_single_chi2(ax:plt.Axes, zgrid:np.ndarray, chi2:np.ndarray, zbest:float|None=None, zmark:float|None=None):
    ax.plot(zgrid, chi2, '-', label='chi2')

    if zbest is not None:
        ax.axvline(zbest, label='z_best')

    if zmark is not None:
        ax.axvline(zmark, color='black', label='z_mark')
    
    if (zbest is not None) or (zmark is not None):
        ax.legend()

def pixel_chi2(galaxy:PhotGalaxy, x_idx:int, y_idx:int, zmark:float|None=None) -> plt.Figure:
    zgrid = galaxy.zgrid
    chi2 = galaxy.chi2_reshaped()[y_idx, x_idx, :]

    zbest = galaxy.zbest_reshaped()[y_idx, x_idx]

    fig, ax = plt.subplots()
    _ax_single_chi2(ax, zgrid, chi2, zbest, zmark)

    title = f'ID:{galaxy.id} x={x_idx} y={y_idx}'
    if zmark is not None:
        title += f' zmark={zmark}'
    fig.suptitle(title)

    return fig

def _total_chi2_values(galaxy:PhotGalaxy) -> tuple[np.ndarray, np.ndarray, float]:
    zgrid = galaxy.zgrid

    if galaxy.total_chi2 is None:
        galaxy.calc_zchi2()
    chi2 = galaxy.total_chi2
    zbest = galaxy.zchi2

    return zgrid, chi2, zbest

def total_chi2(galaxy:PhotGalaxy, zmark:float|None=None) -> plt.Figure:
    fig, ax = plt.subplots()
    _ax_single_chi2(ax, *_total_chi2_values(galaxy), zmark)

    title = f'ID:{galaxy.id}'
    if zmark is not None:
        title += f' zmark={zmark}'
    fig.suptitle(title)

    return fig

def views_and_total_chi2(galaxy:PhotGalaxy,
                         normalise_separate:bool=True, show_text:bool=False, config_file:str='config.yml',
                         zmark:float|None=None
                         ) -> plt.Figure:
    mosaic = """
    SIR
    BBB
    CCC
    """
    fig, axs = plt.subplot_mosaic(mosaic, height_ratios=[1, 0.05, 1.5])

    views._ax_segmap(axs['S'], galaxy)
    views._ax_rgb(axs['I'], galaxy, normalise_separate, config_file)
    views._ax_redshift(axs['R'], galaxy, show_text)

    fig.colorbar(axs['R'].get_images()[0], cax=axs['B'], orientation='horizontal')

    _ax_single_chi2(axs['C'], *_total_chi2_values(galaxy), zmark)

    fig.tight_layout()
    fig.set_figheight(1.5 * fig.get_figheight())

    title = f'ID:{galaxy.id}'
    if zmark is not None:
        title += f' zmark={zmark}'
    fig.suptitle(title)

    return fig

