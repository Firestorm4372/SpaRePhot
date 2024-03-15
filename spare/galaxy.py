import numpy as np

from .data import Data


class Pixel():
    
    def __init__(self, pixel_id:int, galaxy_id:int, filter_values:dict, filter_errors:dict) -> None:
        self.id = pixel_id
        self.galaxy_id = galaxy_id

        self.values = filter_values
        self.errors = filter_errors


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
        ids of each of the `Pixel`s in the galaxy, stored in an array
    pixels : list[Pixel]
        Each of the `Pixel`s in the galaxy. Ordered according to ids
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

        self.pixel_ids = np.arange(self.size, dtype=int).reshape(self.shape)
        self.pixels = self._create_all_pixels_list()

    
    def _create_single_pixel(self, pixel_id:int, x_idx:int, y_idx:int) -> Pixel:
        values = {k: v[y_idx, x_idx] for (k, v) in self.values.items()}
        errors = {k: e[y_idx, x_idx] for (k, e) in self.errors.items()}
        return Pixel(pixel_id, self.id, values, errors)
    
    def _create_all_pixels_list(self) -> list[Pixel]:
        pixels = []

        for y_idx, row in enumerate(self.pixel_ids):
            for x_idx, pixel_id in enumerate(row):
                pix = self._create_single_pixel(pixel_id, x_idx, y_idx)
                pixels.append(pix)

        return pixels


def extract_galaxy(id: int, data:Data, border:int=1) -> Galaxy:
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
 