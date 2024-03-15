import os

import json
import numpy as np

from .filemanage import OverManage
from .data import Data
from .galaxy import Galaxy
from . import galaxy

class Prep():
    
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
        selection = self._galaxy_selection(ids, border)
        run_id = self._generate_run(name, len(selection))
        self._save_selection(selection, run_id)

