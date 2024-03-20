import os

import numpy as np
import eazy
import eazy.hdf5

from ..filemanage import RunManager


__all__ = ['WrapperEAZY', 'init_wrapper_from_hdf5']


class WrapperEAZY():
    """
    Helper class for running EAZY on the EAZY_input.csv file.

    Parameters
    ----------
    run_id : int
        Identifier of the run to extract from
    config_file : str, default 'config.yml'
        Config file to use
    """

    def __init__(self, run_id:int, config_file:str='config.yml') -> None:
        self.run_id = run_id
        self.runmanage = RunManager(config_file)

        self.run_folder = self.runmanage.run_folder(run_id)
        self.eazy_out_folder = f'{self.run_folder}/eazy'

        self.photoz = None

    
    def init_photoz(self, add_params:dict|None=None, param_file:str|None=None, translate_file:str='eazy_files/z_phot.translate') -> None:
        """
        Initialise the photoz object from eazy, using given parameters and translate.
        A set of default parameters are applied if no param_file is given.
        Additional parameters above the param file and defaults can be set with add_params.

        Parameters
        ----------
        add_params : dict | None, default None
            If set, will include these additional parameters
        param_file : str | None, default None
            If set, uses this param_file
        translate_file : str, default 'eazy_files/z_phot.translate'
            Translate file for use in photoz initialisation
        """
        
        # 'default' params if no param_file given
        if param_file is None:
            params = {
                'CATALOG_FILE': f'{self.run_folder}/EAZY_input.csv',
                'CATALOG_FORMAT': 'csv',
                'FILTERS_RES': 'eazy_files/FILTER.RES.latest',
                'TEMPLATES_FILE': 'templates/JADES/JADES_fsps_local.param',
                'APPLY_PRIOR': 'n',
                'Z_MIN': 0.,
                'Z_MAX': 20.,
                'Z_STEP': 0.01,
                'OUTPUT_DIRECTORY': self.eazy_out_folder,
                'MAIN_OUTPUT_FILE': f'{self.eazy_out_folder}/photoz'
            }
        else:
            params = dict()
        
        # add in add_params if needed
        if add_params is not None:
            params |= add_params
        
        # create photoz object
        self.photoz = eazy.photoz.PhotoZ(param_file=param_file, params=params, translate_file=translate_file)

    def run_EAZY_fit(self, save_to_hdf5:bool=True) -> None:
        """
        Run EAZY using the current photoz object.
        Saves instance to hdf5 file.

        Parameters
        ----------
        save_to_hdf5 : bool, default True
            Controls whether photoz object is saved using hdf5
        """
        
        if self.photoz is None:
            raise Exception('Need to init photoz object')

        self.photoz.fit_catalog()
        
        if save_to_hdf5:
            os.makedirs(self.eazy_out_folder, exist_ok=True)
            eazy.hdf5.write_hdf5(self.photoz, f'{self.eazy_out_folder}/photoz.h5')

    def save_EAZY_data(self, folder:str|None=None) -> None:
        """
        Saves relevant EAZY data from the fit as npz file.
        Saved in same location as hdf5 if no location specified.

        Parameters
        ----------
        folder : str | None, default None
            If specified, will save in that folder instead
        """

        if self.photoz is None:
            raise Exception('Must have photoz object instantiated')
        if not np.any(self.photoz.zbest):
            print('Warning: All zbest values are zero')

        fit_data = {
            'zgrid': self.photoz.zgrid,
            'zbest': self.photoz.zbest,
            'chi2': self.photoz.chi2_fit
        }

        if folder is None:
            folder = self.eazy_out_folder

        np.savez(f'{folder}/fit_data.npz', **fit_data)


    def init_and_run_EAZY(self, save_output:bool=True, add_params:dict|None=None, param_file:str|None=None, translate_file:str='eazy_files/z_phot.translate') -> None:
        """
        Perform both the initialisation of photoz object and run of EAZY.

        Initialise the photoz object from eazy, using given parameters and translate.
        A set of default parameters are applied if no param_file is given.
        Additional parameters above the param file and defaults can be set with add_params.

        Parameters
        ----------
        save_output : bool, default True
            Controls if output from the run is saved
        add_params : dict | None, default None
            If set, will include these additional parameters
        param_file : str | None, default None
            If set, uses this param_file
        translate_file : str, default 'eazy_files/z_phot.translate'
            Translate file for use in photoz initialisation
        """

        self.init_photoz(add_params, param_file, translate_file)
        self.run_EAZY_fit(save_to_hdf5=save_output)
        if save_output:
            self.save_EAZY_data()



def init_wrapper_from_hdf5(run_id:int, config_file:str='config.yml') -> WrapperEAZY:
    """
    Initialise a `WrapperEAZY` object from the hdf5 produced after a run.

    Parameters
    ----------
    run_id : int
        Identifier of the run to extract from
    config_file : str, default 'config.yml'
        Config file to use
    """

    wrapper = WrapperEAZY(run_id, config_file)
    wrapper.photoz = eazy.hdf5.initialize_from_hdf5(f'{wrapper.eazy_out_folder}/photoz.h5')
    return wrapper

