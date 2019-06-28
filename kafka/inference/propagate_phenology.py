import numpy as np
import scipy.sparse as sp
import logging

# Set up logging
LOG = logging.getLogger(__name__+".linear_kf")

class TrajectoryFromPrior(object):
    def __init__(self, dates, prior, param_loc, n_params, transformed=True):
        self. prior = prior
        self.param_loc = param_loc
        self.n_params = n_params
        self.dates = dates
        self.transform = transformed
        # I should pass a time grid and interpolate on to that
        # gradient assumes unit separation
        # I'm currently assuming dates matches the time grid and is
        # uniform.
        self.diff = {d: (j - i) for i, j, d in zip(prior[:-1], prior[1:], dates[1:])}
        self.diff[dates[0]] = 0
        self.ratio = {d: (j/i) for i, j, d in zip(prior[:-1], prior[1:], dates[1:])}
        self.ratio[dates[0]] = 1

    def M_matrix(self, date, x_analysis):#, param_loc, n_params):
        prop_matrix_diag = np.ones(x_analysis.shape)
        if self.transform:
            prop_matrix_diag[self.param_loc::self.n_params] = self.ratio[date.date()]
        else:
            prop_matrix_diag[self.param_loc::self.n_params] = \
                self.diff[date.date()]/x_analysis[self.param_loc::self.n_params] + 1

        prop_matrix = sp.csr_matrix((len(x_analysis), len(x_analysis)))
        prop_matrix.setdiag(prop_matrix_diag)
        return prop_matrix


class TrajectoryScaleFromPrior(object):
    def __init__(self, dates, prior, param_loc, n_params, transformed=False):
        """
        Assumes dates matches the time grid and time grid is uniform.
        :param dates:       list of dates for the prior values
        :param prior:       time series of prior values. If they are in transformed space
                            use transformed=True
        :param param_loc:   Location of the parameter to be propagated (LAI) in the parameter vector
        :param n_params:    Number of state parameters per pixel
        :param transformed: If the input prior is in transformed space already then True,
                            otherwise False
        """
        self. prior = prior
        self.param_loc = param_loc
        self.n_params = n_params
        self.dates = dates
        # ToDo:
        # Should I pass a time grid and interpolate on to that or, as is, assume
        # the prior is on the time grid already?
        # ToDo:
        # This assumes a uniform time grid.

        if transformed:
            self.exponent = {d: (np.log(j) - np.log(i))/np.log(i) for i, j, d in zip(prior[:-1], prior[1:], dates[1:])}
        else:
            self.exponent = {d: (j - i)/i for i, j, d in zip(prior[:-1], prior[1:], dates[1:])}
        self.exponent[dates[0]] = 0

    def M_matrix(self, date, x_analysis):#, param_loc, n_params):

        prop_matrix_diag = np.ones(x_analysis.shape)
        # We are in real coordiantes. This is the scale factor for propagation if
        # we want the proportional change in the propagated timestep to equal the proportional
        # change in the prior
        try:
            prop_matrix_diag[self.param_loc::self.n_params] = \
                x_analysis[self.param_loc::self.n_params]**self.exponent[date.date()]
        except KeyError:
            LOG.warning("{} not found in trajectory model. Default to identity".format(date.date()))
        # sometimes x_analysis can go negative. This is unphysical and cannot be transformed
        # back into real space. This leads to nan in the propagation matrix. The propagation
        # is also meaningless at this stage. As this is usually only a few pixels in the image,
        # in order to allow the retrieval to continue, we revert the trajectory for those pixels
        # to identity (today is like yesterday).
        prop_matrix_diag[np.isnan(prop_matrix_diag)] = 1.0

        prop_matrix = sp.csr_matrix((len(x_analysis), len(x_analysis)))
        prop_matrix.setdiag(prop_matrix_diag)
        return prop_matrix


def main():
    import datetime as dt
    prior_real = [1, 1, 2, 4, 4, 3, 1, 10, 4, 6, 2, 6, 8, 8, 9, 10]
    prior = np.exp([-0.5*p for p in prior_real])
    nparams = 10
    dates = [dt.date(2017, 4, 1)+dt.timedelta(d) for d in range(len(prior))]
    traj = TrajectoryScaleFromPrior(dates, prior, 0, nparams)
    #print(traj.dates, traj.diff)
    date = dt.datetime(2017, 4, 13)

    x_analysis = [(np.exp(-0.5*1.0))]*(1000*nparams) # three pixels each with value 1.0
    print(traj.M_matrix(date, np.array(x_analysis)))


if __name__ == "__main__":
    main()
