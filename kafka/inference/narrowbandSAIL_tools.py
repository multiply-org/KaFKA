import logging
import numpy as np
from .utils import block_diag
import os
import gdal

def sail_prior_values():
    """
    :returns
    -------
    The mean prior vector, covariance and inverse covariance matrices."""

    mean = np.array([2.1, np.exp(-60. / 100.),
                     np.exp(-7.0 / 100.), 0.1,
                     np.exp(-50 * 0.0176), np.exp(-100. * 0.002),
                     np.exp(-4. / 2.), 70. / 90., 0.5, 0.9])
    sigma = np.array([0.01, 0.2,
                              0.01, 0.05,
                              0.01, 0.01,
                              0.50, 0.1, 0.1, 0.1])

    covar = np.diag(sigma ** 2).astype(np.float32)
    inv_covar = np.diag(1. / sigma ** 2).astype(np.float32)
    return mean, covar, inv_covar

class SAILPrior(object):
    def __init__(self, parameter_list, state_mask):
        self.parameter_list = parameter_list
        if isinstance(state_mask, (np.ndarray, np.generic)):
            self.state_mask = state_mask
        else:
            self.state_mask = self._read_mask(state_mask)
            # parameter_list = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm',
            #                 'lai', 'ala', 'bsoil', 'psoil']
            # self.mean = np.array([1.19, np.exp(-14.4/100.),
            # np.exp(-4.0/100.), 0.1,
            # np.exp(-50*0.68), np.exp(-100./21.0),
            # np.exp(-3.97/2.),70./90., 0.5, 0.9])
            # sigma = np.array([0.69, 0.016,
            # 0.0086, 0.1,
            # 1.71e-2, 0.017,
            # 0.20, 0.5, 0.5, 0.5])
            mean, c_prior, c_inv_prior = sail_prior_values()
            self.mean = mean
            self.covar = c_prior
            self.inv_covar = c_inv_prior


    def _read_mask(self, fname):
        """Tries to read the mask as a GDAL dataset"""
        if not os.path.exists(fname):
            raise IOError("State mask is neither an array or a file that exists!")
        g = gdal.Open(fname)
        if g is None:
            raise IOError("{:s} can't be opened with GDAL!".format(fname))
        mask = g.ReadAsArray()
        return mask

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




def propagate_LAI_narrowbandSAIL(x_analysis, P_analysis,
                                     P_analysis_inverse,
                                     M_matrix, Q_matrix,
                                     prior=None, state_propagator=None, date=None):
    ''' Propagate a single parameter and
     set the rest of the parameter propagations to zero
     This should be used with a prior for the remaining parameters'''
    nparameters = 10
    lai_position = 6

    x_prior, c_prior, c_inv_prior = sail_prior_values()

    x_forecast = M_matrix.dot(x_analysis)
    n_pixels = len(x_analysis)//nparameters
    x0 = np.tile(x_prior, n_pixels)
    x0[lai_position::nparameters] = x_forecast[lai_position::nparameters] # Update LAI
    lai_post_cov = P_analysis_inverse.diagonal()[lai_position::nparameters]
    lai_Q = Q_matrix.diagonal()[lai_position::nparameters]

    c_inv_prior_mat = []
    for cov, Q in zip(lai_post_cov, lai_Q):
        # inflate uncertainty
        lai_inv_cov = 1.0/((1.0/cov)+Q)
        little_P_forecast_inverse = c_inv_prior.copy()
        little_P_forecast_inverse[lai_position, lai_position] = lai_inv_cov
        c_inv_prior_mat.append(little_P_forecast_inverse)
    P_forecast_inverse=block_diag(c_inv_prior_mat, dtype=np.float32)
    return x0, None, P_forecast_inverse