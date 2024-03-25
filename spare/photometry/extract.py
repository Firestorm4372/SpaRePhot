import json
import numpy as np
import pandas as pd

from ..filemanage import RunManager
from ..galaxy import PhotGalaxy

__all__ = ['Extract']

class Extract():

    def __init__(self, run_id:int, config_file:str='config.yml') -> None:
        self.run_id = run_id
        self.runmanage = RunManager(config_file)

        self.run_folder = self.runmanage.run_folder(run_id)
        self.eazy_out_folder = f'{self.run_folder}/eazy'

        self.catalog = pd.read_csv(f'{self.run_folder}/EAZY_input.csv')
        self.fit_data = np.load(f'{self.eazy_out_folder}/fit_data.npz')

        self.zgrid = self.fit_data['zgrid']
        self.zbest = self.fit_data['zbest']
        self.chi2 = self.fit_data['chi2']

        self.galaxy_ids = np.unique(self.catalog['galaxy_id'])
        self.galaxy_idxs = np.unique(self.catalog['galaxy_idx'])

        self.galaxies:list[PhotGalaxy]|None = None

    
    def _load_galaxy_data(self, folder:str) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, tuple[tuple]]:
        """
        Loads basic galaxy data

        Parameters
        ----------
        folder : str
            Where to load from

        Returns
        -------
        info : dict
        values : dict[str, ndarray]
        errors : dict[str, ndarray]
        segmap : ndarray
        bbox : tuple[tuple]
        """
        
        with open(f'{folder}/info.txt') as f:
            info = json.load(f)
        
        values = np.load(f'{folder}/values.npz')
        errors = np.load(f'{folder}/errors.npz')
        segmap = np.load(f'{folder}/segmap.npy')

        bbox = ((info['ymin'], info['ymax']), (info['xmin'], info['xmax']))

        return info, values, errors, segmap, bbox
    
    def get_galaxy_slice(self, idx:int) -> slice:
        """
        Return the slice within `catalog` (and equivalently `zbest` and `chi2`) that the galaxy occupies with its pixels

        Parameters
        ----------
        idx : int
            `galaxy_idx` to find slice of.
            Note: this is not the id, rather this only identifies within the run

        Returns
        -------
        galaxy_slice : slice
            The relevant slice within the dataframe
        """

        a = np.nonzero(self.catalog['galaxy_idx'] == idx)[0]
        galaxy_slice = slice(a[0], a[-1]+1)

        return galaxy_slice

    
    def extract_galaxies(self) -> None:
        """
        Extract all galaxies from the run as `PhotGalaxy` objects.
        Placed into `galaxies` attribute
        """

        self.galaxies = []

        for idx in self.galaxy_idxs:
            folder = f'{self.run_folder}/galaxies/{idx}'

            info, values, errors, segmap, bbox = self._load_galaxy_data(folder)

            galaxy_slice = self.get_galaxy_slice(idx)
            zbest = self.zbest[galaxy_slice]
            chi2 = self.chi2[galaxy_slice, :]

            self.galaxies.append(
                PhotGalaxy(
                    info['id'], info['centroid'], bbox,
                    values, errors, segmap,
                    self.zgrid, zbest, chi2
                )
            )
            
            