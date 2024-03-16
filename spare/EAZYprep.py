import os

import json
import numpy as np
import pandas as pd

from .filemanage import OverManage
from .data import Data
from .galaxy import Galaxy
from . import galaxy

class Select():
    
    def __init__(self, config_file:str='config.yml') -> None:
        self.data = Data(config_file)
        self.filemanage = OverManage(config_file)
    
    def _galaxy_selection(self, ids:list[int], border:int=0) -> list[Galaxy]:
        return [galaxy.extract_galaxy(i, self.data, border) for i in ids]

    def _generate_run(self, name:str, num_obj:int) -> int:
        """Create run, returns `run_id`"""
        return self.filemanage.add_run(name, num_obj)

    def _save_selection(self, selection:list[Galaxy], run_id:int) -> None:
        run_folder = self.filemanage.run_folder(run_id)
        galaxies_folder = f'{run_folder}/galaxies'
        os.makedirs(galaxies_folder)
        
        for i, galaxy in enumerate(selection):
            folder = f'{galaxies_folder}/{i}'
            os.makedirs(folder)

            with open(f'{folder}/info.txt', 'w') as f:
                json.dump(galaxy.info_dict(), f)

            np.savez(f'{folder}/values.npz', **galaxy.values)
            np.savez(f'{folder}/errors.npz', **galaxy.errors)
            np.save(f'{folder}/segmap.npy', galaxy.segmap)


    def select_and_save(self, name:str, ids:list[int], border:int=0) -> None:
        """Updates internal attributes with `selection` and `run_id`"""
        self.selection = self._galaxy_selection(ids, border)
        self.run_id = self._generate_run(name, len(self.selection))
        self._save_selection(self.selection, self.run_id)


class FileEAZY():
    """
    Save file for use in EAZY.

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

        for galaxy in self.galaxies:
            id_cols = {
                'galaxy_id': np.full(galaxy.size, galaxy.id, dtype=int),
                'pixel_id': galaxy.pixel_ids_flat,
            }
            cols = id_cols | self.extract_pixel_data(galaxy)
            galaxy_dfs.append(pd.DataFrame(cols))

        return pd.concat(galaxy_dfs)
    
    def save_csv_file(self, filepath:str) -> None:
        df = self._create_dataframe()
        df.to_csv(filepath, index_label='id')


def prep_for_EAZY(name:str, ids:list[int], border:int=0, description:str=None, config_file:str='config.yml') -> str:
    # create galaxy selection
    sel = Select(config_file)
    sel.select_and_save(name, ids, border)

    # save csv for EAZY
    run_folder = sel.filemanage.run_folder(sel.run_id)
    FileEAZY(sel.selection).save_csv_file(f'{run_folder}/EAZY_input.csv')

    if description is not None:
        sel.filemanage.add_run_description(sel.run_id, description)
    
    return run_folder

