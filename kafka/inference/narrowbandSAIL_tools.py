import logging
import gdal
import numpy as np
import os
import scipy

from .utils import block_diag
from .kf_tools import propagate_single_parameter

def sail_prior_values():
    """
    :returns
    -------
    The mean prior vector, covariance and inverse covariance matrices."""
    #parameter_list = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm',
    #                 'lai', 'ala', 'bsoil', 'psoil']
    mean = np.array([2.1, np.exp(-60. / 100.),
                     np.exp(-7.0 / 100.), 0.1,
                     np.exp(-50 * 0.0176), np.exp(-100. * 0.002),
                     np.exp(-4. / 2.), 70. / 90., 0.5, 0.9])
    #sigma = np.array([0.01, 0.2,
    #                          0.01, 0.05,
    #                          0.01, 0.01,
    #                          0.50, 0.1, 0.1, 0.1])
    sigma = np.array([0.01, 0.2,
                              0.01, 0.05,
                              0.01, 0.01,
                              1.0, 0.1, 0.1, 0.1])

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
        mask = g.ReadAsArray().astype(int)
        return mask

    def process_prior ( self, time, inv_cov=True):
        # Presumably, self._inference_prior has some method to retrieve
        # a bunch of files for a given date...
        n_pixels = int(self.state_mask.sum())
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
                                     date=None):
    ''' Propagate a single parameter and
     set the rest of the parameter propagations to zero
     This should be used with a prior for the remaining parameters'''
    nparameters = 10
    lai_position = 6
    try:
        trajectory_matrix = M_matrix(date, x_analysis)
    except TypeError:
        trajectory_matrix = M_matrix

    x_prior, c_prior, c_inv_prior = sail_prior_values()
    return propagate_single_parameter(x_analysis, P_analysis,
                                      P_analysis_inverse,
                                      trajectory_matrix, Q_matrix,
                                      nparameters, lai_position,
                                      x_prior, c_inv_prior)

def propagate_LAI_variableQ(x_analysis, P_analysis,
                                     P_analysis_inverse,
                                     M_matrix, LAI_unc,
                                     date=None):
    ''' Propagate a single parameter and
     set the rest of the parameter propagations to zero
     This should be used with a prior for the remaining parameters'''
    nparameters = 10
    lai_position = 6
    #try:
    if type(M_matrix) is np.ndarray or scipy.sparse.issparse(M_matrix):

        trajectory_matrix = M_matrix
    else:
        trajectory_matrix = M_matrix(date, x_analysis)

    Q_matrix = LAI_unc

    for i in range(lai_position, len(x_analysis), nparameters):
        Q_matrix[i,i] = Q_matrix[i,i]*0.5*x_analysis[i]

    x_prior, c_prior, c_inv_prior = sail_prior_values()
    return propagate_single_parameter(x_analysis, P_analysis,
                                      P_analysis_inverse,
                                      trajectory_matrix, Q_matrix,
                                      nparameters, lai_position,
                                      x_prior, c_inv_prior)