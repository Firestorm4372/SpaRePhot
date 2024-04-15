import numpy as np

from .galaxy import Galaxy


class PhotGalaxy(Galaxy):
    """
    Used to store galaxies after the photometric method has been applied.

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
    """

    def __init__(
        self, id: int, centroid: tuple[float], bbox: np.ndarray,
        values: dict[str,np.ndarray], errors: dict[str, np.ndarray], segmap: np.ndarray,
        EAZY_ids: np.ndarray,
        zgrid: np.ndarray, zbest: np.ndarray, chi2: np.ndarray,
        no_fit_value: float = -1
    ) -> None:
        super().__init__(id, centroid, bbox, values, errors, segmap)

        self.EAZY_ids = np.asarray(EAZY_ids)

        self.zgrid = zgrid

        self.no_fit_mask = (zbest == no_fit_value)

        self.zbest = np.ma.masked_array(zbest, mask=self.no_fit_mask)
        self.chi2 = np.ma.masked_array(
            chi2, mask=np.repeat(
                self.no_fit_mask[:, np.newaxis], chi2.shape[1], axis=1
            )
        )

        self.total_chi2:np.ndarray|None = None
        self.zchi2:float|None = None

    def __repr__(self) -> str:
        string =  super().__repr__()
        return f'Phot{string}'
    
    def EAZY_ids_reshaped(self) -> np.ndarray:
        return self.EAZY_ids.reshape(self.shape)
    
    def zbest_reshaped(self) -> np.ndarray:
        return self.zbest.reshape(self.shape)
    
    def chi2_reshaped(self) -> np.ndarray:
        return self.chi2.reshape([*self.shape, self.chi2.shape[-1]])
    

    def calc_zchi2(self) -> None:
        self.total_chi2 = np.sum(self.chi2, axis=0)
        self.zchi2 = self.zgrid[np.argmin(self.total_chi2)]

    
    def calc_zchi2_pixels(self, pixels: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        """
        Estimate redshift using chi2 method, but only with pixels given

        If no mask supplied, will default to the segmap

        Parameters
        ----------
        pixels : array | None, default None
            The pixels to include in the calculation.
            If not set, will default to using the pixels defined in the segmap

        Returns
        -------
        zchi2 : float
            Value that minimises the total chi2
        total_chi2 : array
            Summed chi squared redshift distribution
        """

        if pixels is None:
            pixels = (self.segmap == self.id)

        chi2_pixels = self.chi2_reshaped()[np.nonzero(pixels)]

        total_chi2 = np.sum(chi2_pixels, axis=0)
        zchi2 = self.zgrid[np.argmin(total_chi2)]

        return zchi2, total_chi2

