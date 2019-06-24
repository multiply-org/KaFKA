import numpy as np
import scipy.sparse as sp


class TrajectoryFromPrior(object):
    def __init__(self, dates, prior, param_loc, n_params, transform=True):
        self. prior = prior
        self.param_loc = param_loc
        self.n_params = n_params
        self.dates = dates
        self.transform = transform
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
    def __init__(self, dates, prior, param_loc, n_params, transform=True):
        self. prior = prior
        self.transform = transform
        self.param_loc = param_loc
        self.n_params = n_params
        self.dates = dates
        # I should pass a time grid and interpolate on to that
        # gradient assumes unit separation
        # I'm currently assuming dates matches the time grid and is
        # uniform.
        self.scale = {d: (j - i)/i for i, j, d in zip(prior[:-1], prior[1:], dates[1:])}
        self.scale[dates[0]] = 0
        self.exponent = {d: (np.log(j) - np.log(i))/np.log(i) for i, j, d in zip(prior[:-1], prior[1:], dates[1:])}
        self.exponent[dates[0]] = 0

    def M_matrix(self, date, x_analysis):#, param_loc, n_params):

        prop_matrix_diag = np.ones(x_analysis.shape)
        if self.transform:
            # assumes transformation y' = exp(-0.5y), i.e. the LAI transformation
            # This is the scale factor for propagation if
            # we want the proportional change in the propagated timestep to equal the proportional
            # change in the prior in real space while working in transformed space.
            prop_matrix_diag[self.param_loc::self.n_params] = x_analysis[self.param_loc::self.n_params] \
                                                              ** self.exponent[date.date()]
        else:
            # We are in real coordiantes. This is the scale factor for propagation if
            # we want the proportional change in the propagated timestep to equal the proportional
            # change in the prior
            prop_matrix_diag[self.param_loc::self.n_params] = self.scale[date.date()] + 1.0

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
