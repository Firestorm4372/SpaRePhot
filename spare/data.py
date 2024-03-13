import yaml
import numpy as np
from astropy.io import fits
from astropy.table import Table

class Paths():
    """
    Contains the various paths to input data.

    Attributes
    ----------
    phot_cat : str
        The photometric catalog
    segmap : str
        The segmentation map
    images : dict[str, str]
        Keys are the different filter names, with each image path within
    """
    def __init__(self, images:dict[str, str], catalogs:dict[str, str], filters:list[str]) -> None:
        catalog_folder = catalogs['folder']
        self.phot_cat = f"{catalog_folder}/{catalogs['phot_cat']}"
        self.segmap   = f"{catalog_folder}/{catalogs['segmap']}"

        image_folder = images['folder']
        filename:str = images['filename']
        self.images:dict[str, str] = dict()
        for filt in filters:
            file = filename.replace('?', filt)
            path = f'{image_folder}/{filt}/{file}'
            self.images[filt] = path

    def __repr__(self) -> str:
        d = {'phot_cat': self.phot_cat,
             'segmap': self.segmap,
             'filters': self.images}
        return d.__repr__()
    

class Images():
    """
    Holds the arrays of the various images

    Attributes
    ----------
    values : dict[str, np.ndarray]
        The values of flux in each filter
    errors : dict[str, np.ndarray]
        The errors in flux in each filter
    """

    def __init__(self, filters:list[str], image_paths:dict[str, str]) -> None:
        self.values:dict[str, np.ndarray] = dict()
        self.errors = dict()

        for filt in filters:
            path = image_paths[filt]

            with fits.open(path) as hdul:
                self.values[filt] = hdul['SCI'].data
                self.errors[filt] = hdul['ERR'].data
            

class Data():
    """
    Extracted data from files

    Attributes
    ----------
    filters : list[str]
        The different filters used in the image
    size_cat : Table
        The SIZE table from the photometric catalog
    segmap : ndarray[int]
        Segmentation map of objects by ID
    images : Images
        Contains the values and errors for each filter of image
    paths : Paths
        Contains the various filepaths
    """

    def __init__(self, config_file:str='config.yml') -> None:
        with open(config_file) as f:
            self.config = yaml.safe_load(f)

        self.filters:list[str] = self.config['filters']

        self.paths = Paths(self.config['images'], self.config['catalogs'], self.filters)

        with fits.open(self.paths.phot_cat) as hdul:
            self.size_cat = Table(hdul['SIZE'].data)
        self.size_cat.add_index('ID')
                
        with fits.open(self.paths.segmap) as hdul:
            self.segmap:np.ndarray = hdul[0].data

        self.images = Images(self.filters, self.paths.images)
        
