"""
Extracted the state propagation bits to individual functions
"""
import logging

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spl

from .utils import block_diag

class NoHessianMethod(Exception):
    """An exception triggered when the forward model isn't able to provide an
    estimation of the Hessian"""
    def __init__(self, message):
        self.message = message


def hessian_correction_pixel(hessian, C_obs_inv, innovation):
    hessian_corr = hessian*C_obs_inv*innovation
    return hessian_corr


def hessian_correction(hessian, R_mat, innovation, mask, state_mask,
                       nparams):
    """Calculates higher order Hessian correction for the likelihood term.
    Needs the GP, the Observational uncertainty, the mask...."""
    if hessian is None:
        # The observation operator does not provide a Hessian method. We just
        # return 0, meaning no Hessian correction.
        return 0.
    C_obs_inv = R_mat.diagonal()[state_mask.flatten()]
    mask = mask[state_mask].flatten()

    little_hess = []
    for i, (innov, C, m, hess) in enumerate(zip(innovation, C_obs_inv, mask, hessian)):
        if not m:
            # Pixel is masked
            hessian_corr = np.zeros((nparams, nparams))
        else:
            # Calculate the Hessian correction for this pixel
            hessian_corr = m * hessian_correction_pixel(hess, C, innov)
        little_hess.append(hessian_corr)

    hessian_corr = block_diag(hessian)
    return hessian_corr


def hessian_correction_multiband(hessians, R_mats, innovations,
                                 masks, state_mask, n_bands, nparams):
    """ Non linear correction for the Hessian of the cost function. This handles
    multiple bands. """
    little_hess_cor = []
    for R, hessian, innovation, mask in zip(
            R_mats, hessians, innovations, masks):
        little_hess_cor.append(hessian_correction(hessian, R, innovation,
                                                  mask, state_mask, nparams))
    hessian_corr = sum(little_hess_cor)
    return hessian_corr


def blend_prior(prior_mean, prior_cov_inverse, x_forecast, P_forecast_inverse):
    """
    combine prior mean and inverse covariance with the mean and inverse covariance
    from the previous timestep as the product of gaussian distributions
    :param prior_mean: 1D sparse array
           The prior mean
    :param prior_cov_inverse: sparse array
           The inverse covariance matrix of the prior
    :param x_forecast:

    :param P_forecast_inverse:
    :return: the combined mean and inverse covariance matrix
    """
    # calculate combined covariance
    combined_cov_inv = P_forecast_inverse + prior_cov_inverse
    b = P_forecast_inverse.dot(prior_mean) + prior_cov_inverse.dot(x_forecast)
    b = b.astype(np.float32)
    # Solve for combined mean
    AI = sp.linalg.splu(combined_cov_inv.tocsc())
    x_combined = AI.solve(b)

    return x_combined, combined_cov_inv


def propagate_and_blend_prior(x_analysis, P_analysis, P_analysis_inverse,
                              M_matrix, Q_matrix, 
                              prior=None, state_propagator=None, date=None):
    """

    :param x_analysis:
    :param P_analysis:
    :param P_analysis_inverse:
    :param M_matrix:
    :param Q_matrix:
    :param prior: dictionay that must contain the key 'function' mapped to a
    function that defines the prior and takes the prior dictionary as an argument
    see tip_prior for example). Other dictionary items are optional arguments for
    the prior.
    :param state_propagator:
    :return:
    """
    if state_propagator is not None:
        x_forecast, P_forecast, P_forecast_inverse = state_propagator(
                     x_analysis, P_analysis, P_analysis_inverse, M_matrix, Q_matrix)
    if prior is not None:
        # Prior should call `process_prior` method of prior object
        # this requires a list of parameters, the date and the state grid (a GDAL-
        # readable file)
        prior_mean, prior_cov_inverse = prior.process_prior(date, inv_cov=True)
    if prior is not None and state_propagator is not None:
        x_combined, combined_cov_inv = blend_prior(prior_mean, prior_cov_inverse,
                                                   x_forecast, P_forecast_inverse)
        return x_combined, None, combined_cov_inv
    elif prior is not None:
        return prior_mean, None, prior_cov_inverse
    elif state_propagator is not None:
        return x_forecast, P_forecast, P_forecast_inverse
    else:
        # Clearly not getting a prior here
        return None, None, None


def propagate_standard_kalman(x_analysis, P_analysis, P_analysis_inverse,
                              M_matrix, Q_matrix,
                              prior=None, state_propagator=None, date=None):
    """Standard Kalman filter state propagation using the state covariance
    matrix and a linear state transition model. This function returns `None`
    for the forecast inverse covariance matrix.

    Parameters
    -----------
    x_analysis : array
        The analysis state vector. This comes either from the assimilation or
        directly from a previoulsy propagated state.
    P_analysis : 2D sparse array
        The analysis covariance matrix (typically will be a sparse matrix).
    P_analysis_inverse : 2D sparse array
        The INVERSE analysis covariance matrix (typically a sparse matrix).
        As this is a Kalman update, you will typically pass `None` to it, as
        it is unused.
    M_matrix : 2D array
        The linear state propagation model.
    Q_matrix: 2D array (sparse)
        The state uncertainty inflation matrix that is added to the covariance
        matrix.

    Returns
    -------
    x_forecast (forecast state vector), P_forecast (forecast covariance matrix)
    and `None`"""

    x_forecast = M_matrix.dot(x_analysis)
    P_forecast = P_analysis + Q_matrix
    return x_forecast, P_forecast, None


def propagate_information_filter_SLOW(x_analysis, P_analysis, P_analysis_inverse,
                                      M_matrix, Q_matrix,
                                      prior=None, state_propagator=None, date=None):
    """Information filter state propagation using the INVERSER state covariance
    matrix and a linear state transition model. This function returns `None`
    for the forecast covariance matrix (as this takes forever). This method is
    based on the approximation to the inverse of the KF covariance matrix.

    Parameters
    -----------
    x_analysis : array
        The analysis state vector. This comes either from the assimilation or
        directly from a previoulsy propagated state.
    P_analysis : 2D sparse array
        The analysis covariance matrix (typically will be a sparse matrix).
        As this is an information filter update, you will typically pass `None`
        to it, as it is unused.
    P_analysis_inverse : 2D sparse array
        The INVERSE analysis covariance matrix (typically a sparse matrix).
    M_matrix : 2D array
        The linear state propagation model.
    Q_matrix: 2D array (sparse)
        The state uncertainty inflation matrix that is added to the covariance
        matrix.

    Returns
    -------
    x_forecast (forecast state vector), `None` and P_forecast_inverse (forecast
    inverse covariance matrix)"""
    logging.info("Starting the propagation...")
    x_forecast = M_matrix.dot(x_analysis)
    n, n = P_analysis_inverse.shape
    S= P_analysis_inverse.dot(Q_matrix)
    A = (sp.eye(n) + S).tocsc()
    P_forecast_inverse = spl.spsolve(A, P_analysis_inverse)
    logging.info("Done with propagation")

    return x_forecast, None, P_forecast_inverse


def propagate_information_filter_approx_SLOW(x_analysis, P_analysis, P_analysis_inverse,
                                 M_matrix, Q_matrix,
                                      prior=None, state_propagator=None, date=None):
    """Information filter state propagation using the INVERSER state covariance
    matrix and a linear state transition model. This function returns `None`
    for the forecast covariance matrix (as this takes forever). This method is
    based on calculating the actual matrix from the inverse of the inverse
    covariance, so it is **SLOW**. Mostly here for testing purposes.

    Parameters
    -----------
    x_analysis : array
        The analysis state vector. This comes either from the assimilation or
        directly from a previoulsy propagated state.
    P_analysis : 2D sparse array
        The analysis covariance matrix (typically will be a sparse matrix).
        As this is an information filter update, you will typically pass `None`
        to it, as it is unused.
    P_analysis_inverse : 2D sparse array
        The INVERSE analysis covariance matrix (typically a sparse matrix).
    M_matrix : 2D array
        The linear state propagation model.
    Q_matrix: 2D array (sparse)
        The state uncertainty inflation matrix that is added to the covariance
        matrix.

    Returns
    -------
    x_forecast (forecast state vector), `None` and P_forecast_inverse (forecast
    inverse covariance matrix)"""

    x_forecast = M_matrix.dot(x_analysis)
    # These is an approximation to the information filter equations
    # (see e.g. Terejanu's notes)
    M = P_analysis_inverse   # for convenience and to stay with
    #  Terejanu's notation
    # Main assumption here is that the "inflation" factor is
    # calculated using the main diagonal of M
    D = 1./(1. + M.diagonal()*Q_matrix.diagonal())
    M = sp.dia_matrix((M.diagonal(), 0), shape=M.shape)
    P_forecast_inverse = M.dot(sp.dia_matrix((D, 0),
                                             shape=M.shape))
    return x_forecast, None, P_forecast_inverse


def propagate_single_parameter(x_analysis, P_analysis, P_analysis_inverse,
                               M_matrix, Q_matrix, n_param, location,
                               x_prior, c_inv_prior):
    """ Propagate a single parameter and
     set the rest of the parameter propagations to the prior.
     This should be used with a prior for the remaining parameters"""
    x_forecast = M_matrix.dot(x_analysis)
    n_pixels = len(x_analysis)//n_param
    x0 = np.tile(x_prior, n_pixels)
    x0[location::n_param] = x_forecast[location::n_param]  # Update LAI
    lai_post_cov = P_analysis_inverse.diagonal()[location::n_param]
    lai_Q = Q_matrix.diagonal()[location::n_param]

    c_inv_prior_mat = []
    for cov, Q in zip(lai_post_cov, lai_Q):
        # inflate uncertainty
        lai_inv_cov = 1.0/((1.0/cov)+Q)
        little_P_forecast_inverse = c_inv_prior.copy()
        little_P_forecast_inverse[location::location] = lai_inv_cov
        c_inv_prior_mat.append(little_P_forecast_inverse)
    P_forecast_inverse=block_diag(c_inv_prior_mat, dtype=np.float32)
    return x0, None, P_forecast_inverse


def no_propagation(x_analysis, P_analysis,
                   P_analysis_inverse,
                   M_matrix, Q_matrix,
                   prior=None, state_propagator=None, date=None):
    """
    THIS PROPAGATOR SHOULD NOT BE USED ANY MORE. It is better to set
    the state_propagator to None and to use the Prior exlicitly.

    THIS IS ONLY SUITABLE FOR BROADBAND SAIL uses TIP prior
    No propagation. In this case, we return the original prior. As the
    information filter behaviour is the standard behaviour in KaFKA, we
    only return the inverse covariance matrix. **NOTE** the input parameters
    are there to comply with the API, but are **UNUSED**.

    Parameters
    -----------
    x_analysis : array
        The analysis state vector. This comes either from the assimilation or
        directly from a previoulsy propagated state.
    P_analysis : 2D sparse array
        The analysis covariance matrix (typically will be a sparse matrix).
        As this is an information filter update, you will typically pass `None`
        to it, as it is unused.
    P_analysis_inverse : 2D sparse array
        The INVERSE analysis covariance matrix (typically a sparse matrix).
    M_matrix : 2D array
        The linear state propagation model.
    Q_matrix: 2D array (sparse)
        The state uncertainty inflation matrix that is added to the covariance
        matrix.

    Returns
    -------
    x_forecast (forecast state vector), `None` and P_forecast_inverse (forecast
    inverse covariance matrix)"""

    x_prior, c_prior, c_inv_prior = tip_prior()
    n_pixels = len(x_analysis)/7
    x_forecast = np.array([x_prior for i in range(n_pixels)]).flatten()
    c_inv_prior_mat = [c_inv_prior for n in range(n_pixels)]
    P_forecast_inverse=block_diag(c_inv_prior_mat, dtype=np.float32)

    return x_forecast, None, P_forecast_inverse
