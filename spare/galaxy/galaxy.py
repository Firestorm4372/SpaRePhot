from typing import Literal
import os

import json
import numpy as np


__all__ = ['Galaxy', 'load_galaxy_from_folder']


class Galaxy():
    """
    Base class for a single galaxy (object)

    Parameters
    ----------
    id : int
        Survey id of the object
    centroid : tuple[float]
        (Y,X) position of the centre of the object
    bbox : array
        Bounding box for the object.
        In the form ((YMIN, YMAX), (XMIN, XMAX))
    values, errors : dict[str, array]
        Value and error images for each of the filters
    segmap : array
        Segmentation image
    shape, size : tuple, int
        Shape and size of all images in the object
    pixel_ids : ndarray
        ids map of the different pixels in the galaxy, increasing first in x (`pixel_ids[0,1]=1` etc)
    """

    def __init__(self, id:int, centroid:tuple[float], bbox:np.ndarray, values:dict[str, np.ndarray], errors:dict[str, np.ndarray], segmap:np.ndarray) -> None:
        self.id = id
        
        self.centroid = centroid
        self.Y, self.X = centroid

        self.bbox = bbox
        self.ymin, self.ymax = bbox[0]
        self.xmin, self.xmax = bbox[1]

        self.values = values
        self.errors = errors
        self.segmap = segmap

        self.shape = segmap.shape
        self.size = segmap.size

        self.pixel_ids_flat = np.arange(self.size, dtype=int)
        self.pixel_ids = self.pixel_ids_flat.reshape(self.shape)

    
    def __repr__(self) -> str:
        string = f'Galaxy: {self.id}, (X,Y)({self.X}, {self.Y}), shape{self.shape}'
        return string
    
    def info_dict(self) -> dict:
        return {
            'id': int(self.id),
            'centroid': self.centroid,
            'xmin': int(self.xmin),
            'xmax': int(self.xmax),
            'ymin': int(self.ymin),
            'ymax': int(self.ymax),
            'shape': self.shape
        }
    

    def replace_unused_with_constant(self, unused:float, replace:float, using:Literal['values', 'errors']='errors', verbose:bool=False) -> None:
        """
        Set pixels that do not have data to a different value.
        e.g. set all `0.0` pixels to `-9999` so they are recognisable and ignored by EAZY.

        Replaces in both values and errors images.

        Parameters
        ----------
        unused : float
            The value that pixels will have when unused, e.g. `0.0` in above
        replace : float
            What to replace unused values with, e.g. `-9999` in above
        using : Literal['values', 'errors'], default errors
            Which images to use to search for the unused value.
        verbose : bool, default False
            Control verbosity as executing
        """
        
        using_images:dict[str, np.ndarray] = getattr(self, using)

        if verbose:
            print('Replacing: ')

        for filt, image in using_images.items():
            if verbose:
                print(filt, end=', ')
            condition = (image == unused)
            self.values[filt] = np.where(condition, replace, self.values[filt])
            self.errors[filt] = np.where(condition, replace, self.errors[filt])
            
        if verbose:
            print('\nDone')


    def save_data(self, folder:str) -> None:
        """
        Save data from galaxy into given folder

        Parameters
        ----------
        folder : str
            Where to save galaxy data to
        """

        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/info.txt', 'w') as f:
                json.dump(self.info_dict(), f)

        np.savez(f'{folder}/values.npz', **self.values)
        np.savez(f'{folder}/errors.npz', **self.errors)
        np.save(f'{folder}/segmap.npy', self.segmap)


    def __key(self) -> tuple:
        return (self.id, *self.centroid, self.shape, *self.values.keys())

    def __hash__(self) -> int:
        return hash(self.__key())
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Galaxy):
            return self.__key() == other.__key()
        else:
            return False


def load_galaxy_from_folder(folder:str) -> Galaxy:
    """
    Load galaxy from from given folder path

    Returns
    -------
    galaxy : Galaxy
        Loaded object
    """

    with open(f'{folder}/info.txt') as f:
        info = json.load(f)
    
    values = np.load(f'{folder}/values.npz')
    errors = np.load(f'{folder}/errors.npz')
    segmap = np.load(f'{folder}/segmap.npy')

    bbox = ((info['ymin'], info['ymax']), (info['xmin'], info['xmax']))

    galaxy = Galaxy(info['id'], info['centroid'], bbox, values, errors, segmap)

    return galaxy