import numpy as np

from .phot_galaxy import PhotGalaxy

__all__ = ['ConfGalaxy']


class Confidence():
    """
    Used to hold various attributes and methods only related to confidence

    Option `as_mask` on the is_... functions will invert the array, so values where not true are 1.
    Hence can be used for a masked array
    """

    def __init__(
            self,
            zbest_shaped: np.ndarray, 
            no_fit_mask: np.ndarray,
            lower: np.ndarray, upper: np.ndarray,
            values: list[float],
            interval: float = 2.,
        ) -> None:
        self.lower = lower.reshape(zbest_shaped.shape)
        self.upper = upper.reshape(zbest_shaped.shape)
        self.values = values
        self.interval = interval

        self.difference_unmasked = self.upper - self.lower
        self.difference = np.ma.MaskedArray(self.difference_unmasked, mask=no_fit_mask)
        
        self.zbest = zbest_shaped
    

    def is_above_lower(self, as_mask: bool = False) -> np.ndarray:
        a = (self.zbest >= self.lower)
        if as_mask: return np.logical_not(a)
        else: return a
    
    def is_below_upper(self, as_mask: bool = False) -> np.ndarray:
        a = (self.zbest <= self.upper)
        if as_mask: return np.logical_not(a)
        else: return a

    def is_within(self, as_mask: bool = False) -> np.ndarray:
        a = np.logical_and(self.is_below_upper(), self.is_above_lower())
        if as_mask: return np.logical_not(a)
        else: return a

    def is_confident(self, as_mask: bool = False) -> np.ndarray:
        a = (self.difference <= self.interval)
        if as_mask: return np.logical_not(a)
        else: return a

    def is_within_and_confident(self, as_mask: bool = False) -> np.ndarray:
        a = np.logical_and(self.is_within(), self.is_confident())
        if as_mask: return np.logical_not(a)
        else: return a

    
class ConfGalaxy(PhotGalaxy):
    """
    All feature of `PhotGalaxy`, but also including confidence maps etc

    `zbest` and `chi2` should be passed as 1D and 2D arrays respective.
    Later have functions to produced reshaped.

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

    EAZY_ids : 1D array
        ids of each pixel within the EAZY run the galaxy has been extracted from
    zgrid : 1D array
        The zgrid used by the EAZY run
    zbest: 1D array
        The values of `zbest` produced by EAZY for each of the pixels.
        Pixels where the fit fails are masked out
    chi2 : 2D array
        Array of each of the chi squared distributions for each of the pixels.
        Pixels where the fit fails are masked out

    lower_confidence, upper_confidence: 1D array
        The redshift values of the lower and upper confidence levels respective.
        Unshaped
    confidence_values: list[float]
        Two items in list, the percentile values of the lower and upper respective
    confidence_interval: float, default 2.
        Value required to have between upper and lower to be 'confident' in a z result

    no_fit_value : float, default -1
        The value assigned to a pixels `zbest` when the EAZY fit fails

    Attributes (additional)
    ----------
    shape, size : tuple, int
        Shape and size of all images in the object
    pixel_ids : ndarray
        ids map of the different pixels in the galaxy, increasing first in x (`pixel_ids[0,1]=1` etc)

    no_fit_mask : 1D ndarray
        Mask used to signal that pixels failed to fit with EAZY
    total_chi2 : 1D ndarray | None, default None
        The chi2 distribution produce from summing all pixels in the galaxy.
        Calculated using method `calc_zchi2`
    zchi2 : float | None, default None
        The redshift at the minimum of the total chi2 distribution.
        Calculated using method `calc_zchi2`

    confidence : Confidence
        `Confidence` object to manage most of the confidence specfic attributes and methods
    """

    def __init__(
            self, id: int, centroid: tuple[float], bbox: np.ndarray,
            values: dict[str, np.ndarray], errors: dict[str, np.ndarray], segmap: np.ndarray,
            EAZY_ids: np.ndarray,
            zgrid: np.ndarray, zbest: np.ndarray, chi2: np.ndarray,
            lower_confidence: np.ndarray, upper_confidence: np.ndarray,
            confidence_values: list[float],
            confidence_interval: float = 2.,
            no_fit_value: float = -1,
        ) -> None:
        super().__init__(id, centroid, bbox, values, errors, segmap,
                         EAZY_ids, zgrid, zbest, chi2, no_fit_value)

        self.confidence = Confidence(
            self.zbest_reshaped(), self.no_fit_mask,
            lower_confidence, upper_confidence,
            confidence_values, confidence_interval)
        
    def __repr__(self) -> str:
        string = super().__repr__()
        return f'Conf{string[4:]}'
    
