from typing import Literal

import numpy as np

from .data import Data


class Galaxy():
    """
    Holder for a single galaxy (object)

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
 