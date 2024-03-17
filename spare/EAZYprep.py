from typing import Literal
import os

import json
import numpy as np
import pandas as pd

from .filemanage import OverManage
from .data import Data
from .galaxy import Galaxy
from . import galaxy


class SelectGalaxies():
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
    filemanage : OverManage
    galaxies : list[Galaxy]
        Selected galaxies
    run_id : int | None
        run id the galaxies have been saved under, `None` if unset

    Methods
    -------
    save_selection
        Save the current selection under a new run
    """
    
    def __init__(self, ids:list[int], border:int=0, config_file:str='config.yml') -> None:
        self.data = Data(config_file)
        self.filemanage = OverManage(config_file)

        self.galaxies = [galaxy.extract_galaxy(i, self.data, border) for i in ids]
        self.run_id = None

    def _generate_run(self, name:str) -> int:
        """Create run for current selection, updates `run_id` attribute"""
        self.run_id = self.filemanage.add_run(name, len(self.galaxies))

    def save_selection(self, name:str) -> None:
        self._generate_run(name)
        run_folder = self.filemanage.run_folder(self.run_id)

        galaxies_folder = f'{run_folder}/galaxies'
        os.makedirs(galaxies_folder)
        
        for i, galaxy in enumerate(self.galaxies):
            folder = f'{galaxies_folder}/{i}'
            os.makedirs(folder)

            with open(f'{folder}/info.txt', 'w') as f:
                json.dump(galaxy.info_dict(), f)

            np.savez(f'{folder}/values.npz', **galaxy.values)
            np.savez(f'{folder}/errors.npz', **galaxy.errors)
            np.save(f'{folder}/segmap.npy', galaxy.segmap)


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


def prep_for_EAZY(
        name:str, ids:list[int], border:int=0,
        replace_unused:bool=False, unused:float|None=None, replace:float|None=None, using:Literal['values', 'errors']='errors', verbose_replace:bool=False,
        description:str|None=None, config_file:str='config.yml'
    ) -> str:
    """
    Create and save all data for an EAZY run.

    Parameters
    ----------
    name : str
        Name to give the run
    ids : list[int]
        ids of the galaxies to select
    border : int, default 0
        Size of selected pixels beyond the segmap range
    replace_unused : bool, default False
        Controls whether to replace 'unused' pixel values with a constant.
        Both `unused` and `replace` must also then be set
    unused : float | None, default None
            The value that pixels will have when unused, e.g. `0.0` in above.
    replace : float | None, default None
        What to replace unused values with, e.g. `-9999` in above
    using : Literal['values', 'errors'], default errors
        Which images to use to search for the unused value.
    verbose_replace : bool, default False
        Control verbosity as executing replace
    description : str | None, default None
        Optional description to add to the run
    config_file : str, default config.yml
        Config file to be used
    """

    # create galaxy selection
    selection = SelectGalaxies(ids, border, config_file)

    if replace_unused:
        assert unused is not None
        assert replace is not None
        for gal in selection.galaxies:
            gal.replace_unused_with_constant(unused, replace, using, verbose_replace)

    selection.save_selection(name)

    # save csv for EAZY
    run_folder = selection.filemanage.run_folder(selection.run_id)
    FileEAZY(selection.galaxies).save_csv_file(f'{run_folder}/EAZY_input.csv')

    if description is not None:
        selection.filemanage.add_run_description(selection.run_id, description)
    
    return run_folder

