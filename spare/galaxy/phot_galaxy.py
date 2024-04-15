import numpy as np

from .galaxy import Galaxy


class PhotGalaxy(Galaxy):
    """
    zbest and chi2 should be passed as 1D and 2D arrays respective
    Later have functions to produced reshaped
    """

    def __init__(
        self, id:int, centroid:tuple[float], bbox:np.ndarray,
        values:dict[str,np.ndarray], errors:dict[str, np.ndarray], segmap:np.ndarray,
        EAZY_ids:np.ndarray,
        zgrid:np.ndarray, zbest:np.ndarray, chi2:np.ndarray,
        no_fit_value:float=-1
    ) -> None:
        super().__init__(id, centroid, bbox, values, errors, segmap)

        self.EAZY_ids = EAZY_ids

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

    
    def calc_zchi2_pixels(self, pixels:np.ndarray|None=None) -> tuple[float, np.ndarray]:
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

