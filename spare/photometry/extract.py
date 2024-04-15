import json
import numpy as np
import pandas as pd

from . import run
from ..filemanage import RunManager
from ..galaxy import PhotGalaxy, ConfGalaxy

__all__ = ['Extract']

class Extract():

    def __init__(self, run_id: int, config_file: str = 'config.yml') -> None:
        self.run_id = run_id
        self.config_file = config_file
        self.runmanage = RunManager(config_file)

        self.run_folder = self.runmanage.run_folder(run_id)
        self.eazy_out_folder = f'{self.run_folder}/eazy'

        self.catalog = pd.read_csv(f'{self.run_folder}/EAZY_input.csv')
        self.fit_data = np.load(f'{self.eazy_out_folder}/fit_data.npz')

        self.zgrid = self.fit_data['zgrid']
        self.zbest = self.fit_data['zbest']
        self.chi2 = self.fit_data['chi2']

        self.ids = np.asarray(self.catalog['id'])
        self.galaxy_ids = np.unique(self.catalog['galaxy_id'])
        self.galaxy_idxs = np.unique(self.catalog['galaxy_idx'])

        self.galaxies: list[PhotGalaxy | ConfGalaxy] | None = None

    
    def _load_galaxy_data(self, folder: str) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, tuple[tuple]]:
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
    
    def get_galaxy_slice(self, idx: int) -> slice:
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

    
    def _extract_galaxies_phot(self) -> None:
        self.galaxies = []

        for idx in self.galaxy_idxs:
            folder = f'{self.run_folder}/galaxies/{idx}'

            info, values, errors, segmap, bbox = self._load_galaxy_data(folder)

            galaxy_slice = self.get_galaxy_slice(idx)
            EAZY_ids = self.ids[galaxy_slice]
            zbest = self.zbest[galaxy_slice]
            chi2 = self.chi2[galaxy_slice, :]

            self.galaxies.append(
                PhotGalaxy(
                    id=info['id'], centroid=info['centroid'], bbox=bbox,
                    values=values, errors=errors, segmap=segmap,
                    EAZY_ids=EAZY_ids,
                    zgrid=self.zgrid, zbest=zbest, chi2=chi2
                )
            )

    def extract_galaxies(
            self,
            use_conf_galaxies: bool = False,
            no_fit_value: float = -1., 
            percentiles: list[float] = [16, 84], confidence_interval: float = 2.
        ) -> None:
        """
        Extract all galaxies from the run as either `PhotGalaxy` or `ConfGalaxy` objects.
        Placed into `galaxies` attribute
        """

        self.galaxies = []

        if use_conf_galaxies:
            wrapper = run.init_wrapper_from_hdf5(self.run_id, self.config_file)
            confidence_limits = wrapper.photoz.pz_percentiles(percentiles)
            del wrapper

        for idx in self.galaxy_idxs:
            folder = f'{self.run_folder}/galaxies/{idx}'

            info, values, errors, segmap, bbox = self._load_galaxy_data(folder)

            galaxy_slice = self.get_galaxy_slice(idx)
            EAZY_ids = self.ids[galaxy_slice]
            zbest = self.zbest[galaxy_slice]
            chi2 = self.chi2[galaxy_slice, :]

            if not use_conf_galaxies:
                galaxy = PhotGalaxy(
                    info['id'], info['centroid'], bbox,
                    values, errors, segmap,
                    EAZY_ids,
                    self.zgrid, zbest, chi2,
                    no_fit_value = no_fit_value
                )
            else:
                lower = confidence_limits[galaxy_slice, 0]
                upper = confidence_limits[galaxy_slice, 1]

                galaxy = ConfGalaxy(
                    info['id'], info['centroid'], bbox,
                    values, errors, segmap,
                    EAZY_ids,
                    self.zgrid, zbest, chi2,
                    lower, upper,
                    percentiles, confidence_interval,
                    no_fit_value = no_fit_value
                )

            self.galaxies.append(galaxy)
        
