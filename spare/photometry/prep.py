import os

import numpy as np
import pandas as pd

from ..filemanage import RunManager
from ..galaxy import Galaxy


__all__ = ['SelectionGalaxies', 'FileEAZY']


class SelectionGalaxies():
    """
    Galaxy selection from ids

    Parameters
    ----------
    ids : list[int]
        ids of the galaxies to select
    border : int, default 0
        Size of selected pixels beyond the segmap range
    config_file : str, default config.yml
        Config file to be used

    Attributes
    ----------
    data : Data
    runmanage : RunManager
    galaxies : list[Galaxy]
        Selected galaxies
    run_id : int | None
        run id the galaxies have been saved under, `None` if unset

    Methods
    -------
    save_selection
        Save the current selection under a new run
    """
    
    def __init__(self, galaxies:list[Galaxy], config_file:str='config.yml') -> None:
        self.runmanage = RunManager(config_file)

        self.galaxies = galaxies
        self.run_id = None

    def _generate_run(self, name:str) -> int:
        """Create run for current selection, updates `run_id` attribute"""
        self.run_id = self.runmanage.add_run(name, len(self.galaxies))

    def save_selection(self, name:str) -> None:
        self._generate_run(name)
        run_folder = self.runmanage.run_folder(self.run_id)

        galaxies_folder = f'{run_folder}/galaxies'
        os.makedirs(galaxies_folder)
        
        for i, galaxy in enumerate(self.galaxies):
            folder = f'{galaxies_folder}/{i}'
            galaxy.save_data(folder)


class FileEAZY():
    """
    Used to produce save file for use in EAZY.

    Parameters
    ----------
    selection : list[Galaxy]
        The selection to produce file for
    
    Attributes
    ----------
    galaxies : list[Galaxy]
        Galaxies (selection) to produce file for

    df Columns
    ----------
    id : int
        Unique only to this run, assigned from 0
    galaxy_id : int
        Unique to each detection by JADES, id assigned in segmap
    pixel_id : int
        The id used to identify the pixel in the image, unique to the run, as border of `Galaxy` can vary
    filters, errors : float
        In the form F070W, E070W. Filter values and errors
    """

    def __init__(self, selection:list[Galaxy]) -> None:
        self.galaxies = selection


    @staticmethod
    def extract_pixel_data(galaxy:Galaxy) -> dict[str, np.ndarray]:
        """
        Given a galaxy, will return a dict each filter and error values in ordered array to pixel_ids.
        Note, this expects filter names in the format FXXXX, will then convert errors to EXXXX

        Parameters
        ----------
        galaxy : Galaxy
            Input galaxy object to extract data

        Returns
        -------
        pixel_data : dict[str, ndarray]
            Key value pairs of filter or error name, and 1D array of values in each pixel
        """

        values = {name: image.flatten() for (name, image) in galaxy.values.items()}
        errors = {f'E{name[1:]}': image.flatten() for (name, image) in galaxy.errors.items()}
        return values | errors
    

    def _create_dataframe(self) -> pd.DataFrame:
        galaxy_dfs = []

        for idx, galaxy in enumerate(self.galaxies):
            id_cols = {
                'galaxy_idx': np.full(galaxy.size, idx, dtype=int),
                'galaxy_id': np.full(galaxy.size, galaxy.id, dtype=int),
                'pixel_id': galaxy.pixel_ids_flat,
            }
            cols = id_cols | self.extract_pixel_data(galaxy)
            galaxy_dfs.append(pd.DataFrame(cols))

        df = pd.concat(galaxy_dfs, ignore_index=True)

        return df
    
    def save_csv_file(self, filepath:str) -> None:
        df = self._create_dataframe()
        df.to_csv(filepath, index_label='id')

