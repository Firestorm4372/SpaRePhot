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
        zgrid:np.ndarray, zbest:np.ndarray, chi2:np.ndarray
    ) -> None:
        super().__init__(id, centroid, bbox, values, errors, segmap)

        self.zgrid = zgrid
        self.zbest = zbest
        self.chi2 = chi2
        # self.zbest_shape = zbest.reshape(self.shape)
        # self.chi2_shape = chi2.reshape([*self.shape, self.chi2.shape[-1]])

        self.total_chi2:np.ndarray|None = None
        self.zchi2:float|None = None

    def __repr__(self) -> str:
        string =  super().__repr__()
        return f'Phot{string}'
    

    def calc_zchi2(self) -> None:
        self.total_chi2 = np.sum(self.chi2, axis=0)
        self.zchi2 = self.zgrid[np.argmin(self.total_chi2)]

