from typing import Literal

import numpy as np

from .filemanage import Data
from .galaxy import Galaxy
from .EAZYprep import SelectionGalaxies, FileEAZY
from .EAZYrun import WrapperEAZY


__all__ = ['random_id', 'extract_galaxy', 'prep_for_EAZY', 'run_on_galaxies']


def random_id(data:Data) -> int:
    """
    Return a random id from the dataset.

    Parameters
    ----------
    data : Data
        `Data` object to extract id from

    Returns
    -------
    id : int
        Randomly chosen id from the dataset
    """

    id = data.size_cat.iloc[np.random.randint(len(data.size_cat))]['ID']
    return int(id)

def extract_galaxy(id: int, data:Data, border:int=0) -> Galaxy:
    """
    Return the galaxy object specified, with border specifying extra pixels around the segmap

    Parameters
    ----------
    id : int
        JADES ID of the object
    data : Data
        `Data` object with relevant images
    border : int, default 0
        Number of extra pixels around the segmap to include

    Returns
    -------
    galaxy : Galaxy
        Created `Galaxy` object
    """

    # get relevant data from size_cat
    x = data.size_cat.loc[id]['X']
    y = data.size_cat.loc[id]['Y']
    xmin = data.size_cat.loc[id]['BBOX_XMIN']
    xmax = data.size_cat.loc[id]['BBOX_XMAX']
    ymin = data.size_cat.loc[id]['BBOX_YMIN']
    ymax = data.size_cat.loc[id]['BBOX_YMAX']

    centroid = (y, x)
    bbox = np.array([[ymin, ymax], [xmin, xmax]])

    # extract images
    xmin_b = xmin - border
    xmax_b = xmax + border + 1
    ymin_b = ymin - border
    ymax_b = ymax + border + 1
    values = {filt:im[ymin_b:ymax_b, xmin_b:xmax_b] for (filt, im) in data.images.values.items()}
    errors = {filt:im[ymin_b:ymax_b, xmin_b:xmax_b] for (filt, im) in data.images.errors.items()}
    segmap = data.segmap[ymin_b:ymax_b, xmin_b:xmax_b]

    return Galaxy(id, centroid, bbox, values, errors, segmap)


def prep_for_EAZY(
        name:str, ids:list[int], border:int=0,
        replace_unused:bool=False, unused:float|None=None, replace:float|None=None, using:Literal['values', 'errors']='errors', verbose_replace:bool=False,
        description:str|None=None, config_file:str='config.yml'
    ) -> int:
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

    Returns
    -------
    run_id : int
        id to identify the run created
    """

    # create galaxy selection
    galaxies = [extract_galaxy(id, Data(config_file), border) for id in ids]
    selection = SelectionGalaxies(galaxies, config_file)

    if replace_unused:
        assert unused is not None
        assert replace is not None
        for gal in selection.galaxies:
            gal.replace_unused_with_constant(unused, replace, using, verbose_replace)

    selection.save_selection(name)

    # save csv for EAZY
    run_folder = selection.runmanage.run_folder(selection.run_id)
    FileEAZY(selection.galaxies).save_csv_file(f'{run_folder}/EAZY_input.csv')

    # copy over config
    selection.runmanage.make_config_copy(f'{run_folder}/config.yml')

    if description is not None:
        selection.runmanage.add_run_description(selection.run_id, description)
    
    return selection.run_id


def run_on_galaxies(
        name:str, ids:list[int],
        border:int=0,
        replace_unused:bool=False, unused:float|None=None, replace:float|None=None, using:Literal['values', 'errors']='errors', verbose_replace:bool=False,
        add_params:dict|None=None, param_file:str|None=None, translate_file:str='eazy_files/z_phot.translate',
        save_output:bool=True,
        description:str|None=None, config_file:str='config.yml'
    ) -> tuple[int, WrapperEAZY]:
    """
    Will perform a full run with EAZY on the given galaxies.
    Create and save all data for an EAZY run, then do run.
    Default behaviour saves results to run file.

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

    add_params : dict | None, default None
        If set, will include these additional parameters for EAZY run
    param_file : str | None, default None
        If set, uses this param_file for EAZY run
    translate_file : str, default 'eazy_files/z_phot.translate'
        Translate file for use in photoz initialisation

    save_output : bool, True
        Controls if output is saved
        
    description : str | None, default None
        Optional description to add to the run
    config_file : str, default config.yml
        Config file to be used

    Returns
    -------
    run_id : int
        id to identify the run created
    runner : WrapperEAZY
        The `WrapperEAZY` object created in process
    """

    run_id = prep_for_EAZY(name, ids, border, replace_unused, unused, replace, using, verbose_replace, description, config_file)

    runner = WrapperEAZY(run_id, config_file)
    runner.init_and_run_EAZY(save_output, add_params, param_file, translate_file)


    return run_id, runner

