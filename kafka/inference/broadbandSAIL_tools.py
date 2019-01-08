import logging
import gdal
import numpy as np
import os

from .utils import block_diag
from .kf_tools import propagate_single_parameter


def tip_prior_values():
    """The JRC-TIP prior in a convenient function which is fun for the whole
    family. Note that the effective LAI is here defined in transformed space
    where TLAI = exp(-0.5*LAIe).

    Returns
    -------
    The mean prior vector, covariance and inverse covariance matrices."""
    # broadly TLAI 0->7 for 1sigma

    # Jose's params based on Pinty prior with altered LAI
    #sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.5])
    #x0 = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*1)])

    # Pinty prior values (Note his prior is not for transformed LAI
    # he uses LAI = 1.5 +-5).
    sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 1.0])
    x0 = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*1.5)])

    # Prior values for green leaves
    #sigma = np.array([0.14, 0.7, 0.0959, 0.014, 1.5, 0.2, 1.0])
    #x0 = np.array([0.13, 1.0, 0.1, 0.77, 2.0, 0.18, np.exp(-0.5*1.5)])

    # The individual covariance matrix
    little_p = np.diag(sigma**2).astype(np.float32)
    little_p[5, 2] = 0.8862*0.0959*0.2
    little_p[2, 5] = 0.8862*0.0959*0.2

    inv_p = np.linalg.inv(little_p)
    return x0, little_p, inv_p


class JRCPrior(object):
    """Dummpy 2.7/3.6 prior class following the same interface as 3.6 only
    version."""

    def __init__(self, parameter_list, state_mask):
        """It makes sense to have the list of parameters and state mask
        defined at this point, as they won't change during processing."""
        self.mean, self.covar, self.inv_covar = self._tip_prior()
        self.parameter_list = parameter_list
        if isinstance(state_mask, (np.ndarray, np.generic) ):
            self.state_mask = state_mask
        else:
            self.state_mask = self._read_mask(state_mask)

    def _read_mask(self, fname):
        """Tries to read the mask as a GDAL dataset"""
        if not os.path.exists(fname):
            raise IOError("State mask is neither an array or a file that exists!")
        g = gdal.Open(fname)
        if g is None:
            raise IOError("{:s} can't be opened with GDAL!".format(fname))
        mask = g.ReadAsArray()
        return mask


    def _tip_prior(self):
        """The JRC-TIP prior in a convenient function which is fun for the whole
        family. Note that the effective LAI is here defined in transformed space
        where TLAI = exp(-0.5*LAIe).

        Returns
        -------
        The mean prior vector, covariance and inverse covariance matrices."""
        mean, c_prior, c_inv_prior = tip_prior_values()
        self.mean = mean
        self.covar = c_prior
        self.inv_covar = c_inv_prior
        return mean, c_prior, c_inv_prior

    def process_prior ( self, time, inv_cov=True):
        # Presumably, self._inference_prior has some method to retrieve
        # a bunch of files for a given date...
        n_pixels = self.state_mask.sum()
        x0 = np.array([self.mean for i in range(n_pixels)]).flatten()
        if inv_cov:
            inv_covar_list = [self.inv_covar for m in range(n_pixels)]
            inv_covar = block_diag(inv_covar_list, dtype=np.float32)
            return x0, inv_covar
        else:
            covar_list = [self.covar for m in range(n_pixels)]
            covar = block_diag(covar_list, dtype=np.float32)
            return x0, covar


def propagate_LAI_broadbandSAIL(x_analysis, P_analysis,
                                     P_analysis_inverse,
                                     M_matrix, Q_matrix,
                                     prior=None, state_propagator=None, date=None):
    """Propagate a single parameter and
      set the rest of the parameter propagations to the prior.
      This should be used with a prior for the remaining parameters"""
    nparameters = 7
    lai_position = 6
    x_prior, c_prior, c_inv_prior = tip_prior_values()
    return propagate_single_parameter(x_analysis, P_analysis,
                                      P_analysis_inverse,
                                      M_matrix, Q_matrix,
                                      nparameters, lai_position,
                                      x_prior, c_inv_prior)
