import logging
import gdal
import numpy as np
import os
import datetime as dt

from .utils import block_diag
from .narrowbandSAIL_tools import sail_prior_values

# Set up logging
import logging
LOG = logging.getLogger(__name__+".temporal_prior")

'''
def prior_values(time):
    mean, covar, inv_covar = sail_prior_values()
    lai_position = 6

    if time < dt.datetime(2017, 6, 25):
        mean[lai_position] = np.exp(-0.1 / 2.)
    else:
        mean[lai_position] = np.exp(-4. / 2.)

    return mean, covar, inv_covar
'''


class TemporalSAILPrior(object):
    def __init__(self, parameter_list, state_mask, LAI=None, LAI_unc=None):
        self.parameter_list = parameter_list
        if isinstance(state_mask, (np.ndarray, np.generic)):
            self.state_mask = state_mask
        else:
            self.state_mask = self._read_mask(state_mask)
        self.LAI = LAI
        self.LAI_unc = LAI_unc


    def _read_mask(self, fname):
        """Tries to read the mask as a GDAL dataset"""
        if not os.path.exists(fname):
            raise IOError("State mask is neither an array or a file that exists!")
        g = gdal.Open(fname)
        if g is None:
            raise IOError("{:s} can't be opened with GDAL!".format(fname))
        mask = g.ReadAsArray().astype(int)
        return mask

    def process_prior(self, time, inv_cov=True):

        # mean, covar, inv_covar = prior_values(time)
        mean, covar, inv_covar = sail_prior_values()
        lai_position = 6
        try:
            mean[lai_position] = self.LAI[time.date()]
        except KeyError: #use default prior if none in LAI
            LOG.info("No prior LAI for {} using default = {}".format(
                    time.date(), mean[lai_position]))
        if self.LAI_unc is not None:
            try:
                inv_covar[lai_position, lai_position] = self.LAI_unc[time.date()]
            except KeyError:  # use default prior if none in LAI
                LOG.info("No prior LAI_unc for {} using default = {}".format(
                          time.date(), inv_covar[lai_position]))
        n_pixels = self.state_mask.sum()
        x0 = np.array([mean for _ in range(n_pixels)]).flatten()
        if inv_cov:
            inv_covar_list = [inv_covar for _ in range(n_pixels)]
            inv_covar = block_diag(inv_covar_list, dtype=np.float32)
            return x0, inv_covar
        else:
            covar_list = [covar for _ in range(n_pixels)]
            covar = block_diag(covar_list, dtype=np.float32)
            return x0, covar
