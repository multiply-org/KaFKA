#!/usr/bin/env python
"""A fast Kalman filter implementation designed with raster data in mind. This
implementation basically performs a very fast update of the filter."""

# KaFKA A fast Kalman filter implementation for raster based datasets.
# Copyright (c) 2017 J Gomez-Dans. All rights reserved.
#
# This file is part of KaFKA.
#
# KaFKA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KaFKA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KaFKA.  If not, see <http://www.gnu.org/licenses/>.

import logging
from collections import namedtuple

import numpy as np

import scipy.sparse as sp

from .inference import variational_kalman
from .inference import variational_kalman_multiband
from .inference import iterate_time_grid
from .inference.kf_tools import propagate_and_blend_prior

# Set up logging

LOG = logging.getLogger(__name__+".linear_kf")


__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"

Metadata = namedtuple('Metadata', 'mask uncertainty')
Previous_State = namedtuple("Previous_State",
                            "timestamp x_vect cov_m icov_mv")


class LinearKalman (object):
    """The main Kalman filter class operating in raster data sets. Note that the
    goal of this class is not to consider complex, time evolving models, but
    rather grotty "0-th" order models!"""
    def __init__(self, observations, output, state_mask,
                 create_observation_operator, parameters_list,
                 state_propagation=None,
                 linear=True, diagnostics=True, prior=None):
        """The class creator takes (i) an observations object, (ii) an output
        writer object, (iii) the state mask (a boolean 2D array indicating which
        pixels are used in the inference), and additionally, (iv) a state
        propagation scheme (defaults to `propagate_information_filter`),
        whether a linear model is used or not, the number of parameters in
        the state vector, whether diagnostics are being reported, and the
        number of bands per observation.
        """
        self.parameters_list = parameters_list # A list of parameter names
                                     # Required by prior
        self.n_params = len(self.parameters_list)
    
        self.observations = observations
        self.output = output
        self.diagnostics = diagnostics
        self.state_mask = state_mask
        self.n_state_elems = self.state_mask.sum()
        self._state_propagator = state_propagation
        self._advance = propagate_and_blend_prior
        self.prior = prior
        # this allows you to pass additional information with prior needed by
        # specific functions. All priors need a dictionary with ['function'] key.
        # Other keys are optional
        self._create_observation_operator = create_observation_operator
        LOG.info("Starting KaFKA run!!!")

    def advance(self, x_analysis, P_analysis, P_analysis_inverse,
                trajectory_model, trajectory_uncertainty):
        LOG.info("Calling state propagator...")
        x_forecast, P_forecast, P_forecast_inverse = \
            self._advance(x_analysis, P_analysis, P_analysis_inverse,
                          trajectory_model, trajectory_uncertainty,
                          prior=self.prior, date=self.current_timestep,
                          state_propagator=self._state_propagator)

        return x_forecast, P_forecast, P_forecast_inverse

    def _set_plot_view(self, diag_string, timestep, obs):
        """This sets out the plot view for each iteration. Please override this
        method with whatever you want."""
        pass

    def _plotter_iteration_start(self, plot_obj, x, obs, mask):
        """We call this diagnostic method at the **START** of the iteration"""
        pass

    def _plotter_iteration_end(self, plot_obj, x, P, innovation, mask):
        """We call this diagnostic method at the **END** of the iteration"""
        pass

    def set_trajectory_model(self):
        """In a Kalman filter, the state is progated from time `t` to `t+1`
        using a model. We assume that this model is a matrix, and for the time
        being, the matrix is the identity matrix. That's how we roll!"""
        n = self.n_state_elems
        self.trajectory_model = sp.eye(self.n_params*n, self.n_params*n,
                                       format="csr")

    def set_trajectory_uncertainty(self, Q):
        """In a Kalman filter, the model that propagates the state from time
        `t` to `t+1` is assumed to be *wrong*, and this is indicated by having
        additive Gaussian noise, which we assume is zero-mean, and controlled
        by a covariance matrix `Q`. Here, you can provide the main diagonal of
         `Q`.

        Parameters
        -----------
        Q: array
            The main diagonal of the model uncertainty covariance matrix.
        """
        n = self.n_state_elems
        self.trajectory_uncertainty = sp.eye(self.n_params*n, self.n_params*n,
                                             format="csr")
        self.trajectory_uncertainty.setdiag(Q)

    def _get_observations_timestep(self, timestep, band=None):
        """A method that returns the observations, mask and uncertainty for a
        particular timestep. It is envisaged that applications will specialise
        this method to efficiently read and process raster datasets from disk,
        but this version will just select from a numpy array or something.

        Parameters
        ----------
        timestep: int
            This is the time step that we require the information for.
        band: int
            For multiband datasets, this selects the band to use, or `None` if
            single band dataset is used.

        Returns
        -------
        Observations (N*N), uncertainty (N*N) and mask (N*N) arrays, as well
        as relevant metadata
        """
        data = self.observations.get_band_data(timestep, band)
        return (data.observations, data.uncertainty, data.mask,
                data.metadata, data.emulator)

    def run(self, time_grid, x_forecast, P_forecast, P_forecast_inverse,
            diag_str="diagnostics",
            band=None, approx_diagonal=True, refine_diag=True,
            iter_obs_op=False, is_robust=False, dates=None):
        """Runs a complete assimilation run. Requires a temporal grid (where
        we store the timesteps where the inferences will be done, and starting
        values for the state and covariance (or inverse covariance) matrices.

        The time_grid ought to be a list with the time steps given in the same
        form as self.observation_times"""
        for timestep, locate_times in iterate_time_grid(time_grid, self.observations.dates):
            self.current_timestep = timestep

            if len(locate_times) == 0:
                # Just advance the time
                x_analysis = x_forecast
                P_analysis = P_forecast
                P_analysis_inverse = P_forecast_inverse
                LOG.info("No observations in this time")

            else:
                # We do have data, so we assimilate
                x_analysis, P_analysis, P_analysis_inverse = self.assimilate_multiple_bands(
                                     locate_times, x_forecast, P_forecast,
                                     P_forecast_inverse,
                                     approx_diagonal=approx_diagonal,
                                     refine_diag=refine_diag,
                                     iter_obs_op=iter_obs_op,
                                     is_robust=is_robust, diag_str=diag_str)
            LOG.info("Dumping results to disk")
            self.output.dump_data(timestep, x_analysis, P_analysis, P_analysis_inverse, self.state_mask, self.n_params)
            LOG.info("Advancing state, %s" % timestep.strftime("%Y-%m-%d"))
            x_forecast, P_forecast, P_forecast_inverse = self.advance(x_analysis, P_analysis, P_analysis_inverse,
                                                                      self.trajectory_model,
                                                                      self.trajectory_uncertainty)
            self.output.dump_state(timestep, x_forecast, P_forecast, P_forecast_inverse, self.state_mask)


    def assimilate_multiple_bands(self, locate_times, x_forecast, P_forecast,
                   P_forecast_inverse,
                   approx_diagonal=True, refine_diag=False,
                   iter_obs_op=False, is_robust=False, diag_str="diag"):        
        """The method assimilates the observatins at timestep `timestep`, using
        a prior a multivariate Gaussian distribution with mean `x_forecast` and
        variance `P_forecast`. THIS DOES ALL BANDS SIMULTANEOUSLY!!!!!"""
        for step in locate_times:
            LOG.info("Assimilating %s..." % step.strftime("%Y-%m-%d"))
            current_data = []
            # Reads all bands into one list
            for band in range(self.observations.bands_per_observation[step]):
                current_data.append(self.observations.get_band_data(step, 
                                                                    band))

            x_analysis, P_analysis, P_analysis_inverse, innovations = \
                self.do_all_bands(step, current_data, x_forecast, P_forecast,
                                  P_forecast_inverse)
            x_forecast = x_analysis*1.
            try:
                P_forecast = P_analysis*1.
            except:
                P_forecast = None
            try:
                P_forecast_inverse = P_analysis_inverse*1.
            except:
                P_forecast_inverse = None
                        
        return x_analysis, P_analysis, P_analysis_inverse            


    def do_all_bands(self, timestep, current_data, x_forecast, P_forecast,
                        P_forecast_inverse, convergence_tolerance=1e-3,
                        min_iterations=2):
        converged = False
        # Linearisation point is set to x_forecast for first iteration
        x_prev = x_forecast*1.
        n_iter = 1
        n_bands = len(current_data)
        while not converged:
            Y = []
            MASK = []
            UNC = []
            META = []
            H_matrix = []
            for band, data in enumerate(current_data):
                # Create H0 and H_matrix around x_prev
                # Also extract single band information from nice package
                # this allows us to use the same interface as current
                # Deferring processing to a new solver method in solvers.py

                try:
                    H_matrix_= self._create_observation_operator(self.n_params,
                                                         data.emulator,
                                                         data.metadata,
                                                         data.mask,
                                                         self.state_mask,
                                                         x_prev,
                                                         band,
                                                         calc_hess = False)
                except ValueError as e:
                    if (sum(data.mask[self.state_mask])== 0):
                        LOG.error("All observations masked out")
                    raise


                H_matrix.append(H_matrix_)
                Y.append(data.observations)
                MASK.append(data.mask)
                UNC.append(data.uncertainty)
                META.append(data.metadata)
            # Now call the solver 
            x_analysis, P_analysis, P_analysis_inverse, \
                innovations, fwd_modelled = self.solver_multiband(
                    Y, MASK, H_matrix, x_prev, x_forecast,
                    P_forecast, P_forecast_inverse, UNC,
                    META)
            

            # Test convergence. We calculate the l2 norm of the difference
            # between the state at the previous iteration and the current one
            # There might be better tests, but this is quite straightforward
            #passer_mask = data.mask[self.state_mask]
            #maska = np.concatenate([passer_mask.ravel()
            #                        for i in range(self.n_params)])
            #convergence_norm = np.linalg.norm(x_analysis[maska] -
            #                                  x_prev[maska])/float(maska.sum())
            convergence_norm = np.linalg.norm(x_analysis - x_prev)/float(len(x_analysis))
            LOG.info(
                "Band {:d}, Iteration # {:d}, convergence norm: {:g}".format(
                    band, n_iter, convergence_norm))
            if (convergence_norm < convergence_tolerance) and (
                    n_iter >= min_iterations):
                # Converged!
                converged = True
            elif n_iter >= 25:
                # Too many iterations
                LOG.warning("Bailing out after 25 iterations!!!!!!")
                converged = True

            x_prev = x_analysis*1.
            n_iter += 1
            
        # Once we have converged...
        # Correct hessian for higher order terms
        #split_points = [m.sum( ) for m in MASK]
        #todo include this part. Rather than commenting out, we should decide whether to correct or not
        # HESSIAN = []
        # INNOVATIONS = np.split(innovations, n_bands)
        # for band, data in enumerate(current_data):
                # calculate the hessian for the solution
                # _,_,hessian= self._create_observation_operator(self.n_params,
                #                                          data.emulator,
                #                                          data.metadata,
                #                                          data.mask,
                #                                          self.state_mask,
                #                                          x_analysis,
                #                                          band,
                #                                          calc_hess = True)
                # HESSIAN.append(hessian)
        # P_correction = hessian_correction_multiband(HESSIAN,
        #                                             UNC, INNOVATIONS, MASK,
        #                                             self.state_mask, n_bands,
        #                                             self.n_params)
        # P_analysis_inverse = P_analysis_inverse - P_correction
        # Rarely, this returns a small negative value. For now set to nan.
        # May require further investigation in the future
        # P_analysis_inverse[P_analysis_inverse<0] = np.nan

        # Done with this observation, move along...
        
        return x_analysis, P_analysis, P_analysis_inverse, innovations
                
    def assimilate(self, locate_times, x_forecast, P_forecast,
                   P_forecast_inverse,
                   approx_diagonal=True, refine_diag=False,
                   iter_obs_op=False, is_robust=False, diag_str="diag"):
        """The method assimilates the observatins at timestep `timestep`, using
        a prior a multivariate Gaussian distribution with mean `x_forecast` and
        variance `P_forecast`."""
        for step in locate_times:
            LOG.info("Assimilating %s..." % step.strftime("%Y-%m-%d"))
            for band in range(self.observations.bands_per_observation[step]):
                x_analysis, P_analysis, P_analysis_inverse, innovations = \
                    self.assimilate_band(band, step, x_forecast, P_forecast,
                                         P_forecast_inverse)
                # Once the band is assimilated, the posterior (i.e. analysis)
                # becomes the prior (i.e. forecast)
                x_forecast = x_analysis*1.
                if P_analysis is not None:
                    P_forecast = P_analysis*1.
                else:
                    P_forecast = None
                if P_analysis_inverse is not None:
                    P_forecast_inverse = P_analysis_inverse*1.
                else:
                    P_forecast_inverse = None
                #P_forecast_inv = P_analysis_inverse*1.

        self.previous_state = Previous_State(step, x_analysis,
                                             P_analysis, P_analysis_inverse)

        return x_analysis, P_analysis, P_analysis_inverse

    def assimilate_band(self, band, timestep, x_forecast, P_forecast,
                        P_forecast_inverse, convergence_tolerance=1e-3,
                        min_iterations=1):
        """A method to assimilate a band using an interative linearisation
        approach.  This method isn't very sexy, just (i) reads the data, (ii)
        iterates over the solution, updating the linearisation point and calls
        the solver a few times. Most of the work is done by the methods that
        are being called from withing, but the structure is less confusing.
        There are some things missing, such as a "robust" method and I am yet
        to add the correction to the Hessian at the end of the method just
         before it returns to the caller."""

        # Read the relevant data for cufrent timestep and band
        data = self.observations.get_band_data(timestep, band)
        not_converged = True
        # Linearisation point is set to x_forecast for first iteration
        x_prev = x_forecast*1.
        n_iter = 1
        while not_converged:
            # Create H matrix
            H_matrix = self._create_observation_operator(self.n_params,
                                                         data.emulator,
                                                         data.metadata,
                                                         data.mask,
                                                         self.state_mask,
                                                         x_prev,
                                                         band)
            x_analysis, P_analysis, P_analysis_inverse, \
                innovations, fwd_modelled = self.solver(
                    data.observations, data.mask, H_matrix, x_forecast,
                    P_forecast, P_forecast_inverse, data.uncertainty,
                    data.metadata)

            # Test convergence. We calculate the l2 norm of the difference
            # between the state at the previous iteration and the current one
            # There might be better tests, but this is quite straightforward
            passer_mask = data.mask[self.state_mask]
            maska = np.concatenate([passer_mask.ravel()
                                    for i in range(self.n_params)])
            convergence_norm = np.linalg.norm(x_analysis[maska] -
                                              x_prev[maska])/float(maska.sum())
            LOG.info(
                "Band {:d}, Iteration # {:d}, convergence norm: {:g}".format(
                    band, n_iter, convergence_norm))
            if (convergence_norm < convergence_tolerance) and (
                    n_iter >= min_iterations):
                # Converged!
                not_converged = False
            elif n_iter > 25:
                # Too many iterations
                LOG.warning("Bailing out after 25 iterations!!!!!!")
                not_converged = False

            x_prev = x_analysis
            n_iter += 1
        #todo include this part. Rather than commenting out, we should decide whether to correct or not
        # Correct hessian for higher order terms
        # P_correction = hessian_correction(data.emulator, x_analysis,
        #                                   data.uncertainty, innovations,
        #                                   data.mask, self.state_mask, band,
        #                                   self.n_params)
        # UPDATE HESSIAN WITH HIGHER ORDER CONTRIBUTION
        # P_analysis_inverse = P_analysis_inverse - P_correction
        # Rarely, this returns a small negative value. For now set to nan.
        # May require further investigation in the future
        # negative_values = P_analysis_inverse<0
        # if any(negative_values):
        #     P_analysis_inverse[negative_values] = np.nan
        #     LOG.warning("{} negative values in inverse covariance matrix".format(
        #         sum(negative_values)))


        import matplotlib.pyplot as plt
        M = self.state_mask*1.
        M[self.state_mask] = x_analysis[6::7]
        plt.figure()
        plt.imshow(M[650:730, 1180:1280], interpolation="nearest", vmin=0.1, vmax=0.5)
        plt.title("Band: %d, Date:"%band + timestep.strftime("%Y-%m-%d"))
        
        return x_analysis, P_analysis, P_analysis_inverse, innovations

    def solver(self, observations, mask, H_matrix, x_forecast, P_forecast,
               P_forecast_inv, R_mat, the_metadata):

        x_analysis, P_analysis, P_analysis_inv, \
            innovations_prime, fwd_modelled = \
            variational_kalman(
                observations, mask, self.state_mask, R_mat, H_matrix,
                self.n_params,
                x_forecast, P_forecast, P_forecast_inv, the_metadata)

        return x_analysis, P_analysis, P_analysis_inv, \
            innovations_prime, fwd_modelled


    def solver_multiband(self, observations, mask, H_matrix, x0, x_forecast, P_forecast,
               P_forecast_inv, R_mat, the_metadata):

        x_analysis, P_analysis, P_analysis_inv, \
            innovations_prime, fwd_modelled = \
            variational_kalman_multiband(
                observations, mask, self.state_mask, R_mat, H_matrix,
                self.n_params, x0,
                x_forecast, P_forecast, P_forecast_inv, the_metadata)

        return x_analysis, P_analysis, P_analysis_inv, \
            innovations_prime, fwd_modelled
