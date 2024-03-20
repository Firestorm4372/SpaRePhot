import numpy as np

from .galaxy import Galaxy


class PhotGalaxy(Galaxy):
    """
    zbest and chi2 should be passed as 1D and 2D arrays respective
    Will be reshaped by the init
    """

    def __init__(
        self, id:int, centroid:tuple[float], bbox:np.ndarray,
        values:dict[str,np.ndarray], errors:dict[str, np.ndarray], segmap:np.ndarray,
        zgrid:np.ndarray, zbest:np.ndarray, chi2:np.ndarray,
        no_fit_value:float=-1
    ) -> None:
        super().__init__(id, centroid, bbox, values, errors, segmap)

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
    
    def zbest_reshaped(self) -> np.ndarray:
        return self.zbest.reshape(self.shape)
    
    def chi2_reshaped(self) -> np.ndarray:
        return self.chi2.reshape([*self.shape, self.chi2.shape[-1]])
    

    def calc_zchi2(self) -> None:
        self.total_chi2 = np.sum(self.chi2, axis=0)
        self.zchi2 = self.zgrid[np.argmin(self.total_chi2)]

