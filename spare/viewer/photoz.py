import numpy as np
import matplotlib.pyplot as plt

from ..galaxy import PhotGalaxy



def _ax_single_chi2(ax:plt.Axes, zgrid:np.ndarray, chi2:np.ndarray, zbest:float, zmark:float|None=None):
    ax.plot(zgrid, chi2, '-', label='chi2')
    ax.axvline(zbest, label='z_best')
    if zmark is not None:
        ax.axvline(zmark, color='black', label='z_mark')
    ax.legend()

def pixel_chi2(galaxy:PhotGalaxy, x_idx:int, y_idx:int, zmark:float|None=None) -> plt.Figure:
    zgrid = galaxy.zgrid
    chi2 = galaxy.chi2_reshaped()[y_idx, x_idx, :]

    zbest = galaxy.zbest_reshaped()[y_idx, x_idx]

    fig, ax = plt.subplots()
    _ax_single_chi2(ax, zgrid, chi2, zbest, zmark)

    return fig

def total_chi2(galaxy:PhotGalaxy, zmark:float|None=None) -> plt.Figure:
    zgrid = galaxy.zgrid

    if galaxy.total_chi2 is None:
        galaxy.calc_zchi2()
    chi2 = galaxy.total_chi2
    zbest = galaxy.zchi2

    fig, ax = plt.subplots()
    _ax_single_chi2(ax, zgrid, chi2, zbest, zmark)

    return fig