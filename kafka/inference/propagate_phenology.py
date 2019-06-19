import numpy as np
import scipy.sparse as sp


class TrajectoryFromPrior(object):
    def __init__(self, dates, prior):
        self. prior = prior
        #self.param_loc = param_loc
        #self.n_params = n_params
        self.dates = dates
        # I should pass a time grid and interpolate on to that
        # gradient assumes unit separation
        # I'm currently assuming dates matches the time grid and is
        # uniform.
        self.diff = {d: (j - i) for i, j, d in zip(prior[:-1], prior[1:], dates[1:])}
        self.diff[dates[0]] = 0

    def M_matrix(self, date, x_analysis, param_loc, n_params):
        prop_matrix_diag = np.ones(x_analysis.shape)
        prop_matrix_diag[param_loc::n_params] = \
            self.diff[date.date()]/x_analysis[param_loc::n_params] + 1

        prop_matrix = sp.csr_matrix((len(x_analysis), len(x_analysis)))
        prop_matrix.setdiag(prop_matrix_diag)
        return prop_matrix


class TrajectoryScaleFromPrior(object):
    def __init__(self, dates, prior):
        self. prior = prior
        #self.param_loc = param_loc
        #self.n_params = n_params
        self.dates = dates
        # I should pass a time grid and interpolate on to that
        # gradient assumes unit separation
        # I'm currently assuming dates matches the time grid and is
        # uniform.
        self.scale = {d: (j - i)/i for i, j, d in zip(prior[:-1], prior[1:], dates[1:])}
        self.scale[dates[0]] = 0

    def M_matrix(self, date, x_analysis, param_loc, n_params):
        prop_matrix_diag = np.ones(x_analysis.shape)
        prop_matrix_diag[param_loc::n_params] = self.scale[date.date()] + 1.0

        prop_matrix = sp.csr_matrix((len(x_analysis), len(x_analysis)))
        prop_matrix.setdiag(prop_matrix_diag)
        return prop_matrix


def main():
    import datetime as dt
    prior = [1, 1, 2, 4, 4, 3, 1]
    dates = [dt.date(2017, 4, 1)+dt.timedelta(d) for d in range(len(prior))]
    traj = TrajectoryScaleFromPrior(dates, prior)
    #print(traj.dates, traj.diff)
    date = dt.datetime(2017, 4, 7)

    nparams = 2
    x_analysis = [1.0]*(3*nparams) # three pixels each with value 1.0
    print(traj.M_matrix(date, np.array(x_analysis), 0, nparams))


if __name__ == "__main__":
    main()
