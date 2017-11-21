#!/usr/bin/env python
"""Some utility functions used by the main code base."""

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

__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
#import netCDF4
import datetime as dt
import os
import gdal

import logging
LOG = logging.getLogger(__name__)


def iterate_time_grid(time_grid, the_dates):
    is_first = True
    istart_date = time_grid[0]
    for ii, timestep in enumerate(time_grid[1:]):
        # First locate all available observations for time step of interest.
        # Note that there could be more than one...
        locate_times_idx = np.where(np.logical_and(
            np.array(the_dates) >= istart_date,
            np.array(the_dates) < timestep), True, False)
        locate_times = np.array(the_dates)[
                           locate_times_idx]
        current_timestep = timestep
            
        LOG.info("Doing timestep from {} -> {}".format(
                istart_date.strftime("%Y-%m-%d"),
                timestep.strftime("%Y-%m-%d") ))
        LOG.info("# of Observations: %d" % len(locate_times))
        for iobs in locate_times:
                LOG.info("\t->{}".format(iobs.strftime("%Y-%m-%d")))
        istart_date = timestep
        if is_first:
            yield timestep, locate_times, True
            is_first = False
        else:
            yield timestep, locate_times, False


def run_emulator(gp, x, tol=None):
    # We select the unique values in vector x
    # Note that we could have done this using e.g. a histogram
    # or some other method to select solutions "close enough"
    unique_vectors = np.vstack({tuple(row) for row in x})
    if len(unique_vectors) == 1:  # Prior!
        cluster_labels = np.zeros(x.shape[0], dtype=np.int16)
    elif len(unique_vectors) > 1e6:
        
        LOG.info("Clustering parameter space")
        mean = np.mean(x, axis=0) # 7 dimensions
        cov = np.cov(x, rowvar=0) # 4 x 4 dimensions
        # Draw a 300 element LUT
        unique_vectors = np.random.multivariate_normal( mean, cov, 
                                                       5000)
        # Assign each element of x to a LUT/cluster entry
        cluster_labels = locate_in_lut(unique_vectors, x)
    # Runs emulator for emulation subset
    try:
        H_, dH_ = gp.predict(unique_vectors, do_unc=False)
    except ValueError:
        # Needed for newer gp version
        H_, _, dH_ = gp.predict(unique_vectors, do_unc=False)

    H = np.zeros(x.shape[0])
    dH = np.zeros_like(x)
    try:
        nclust = cluster_labels.shape
    except NameError:
        for i, uniq in enumerate(unique_vectors):
            passer = np.all(x == uniq, axis=1)
            H[passer] = H_[i]
            dH[passer, :] = dH_[i, :]
        return H, dH

    for label in np.unique(cluster_labels):
        H[cluster_labels == label] = H_[label]
        dH[cluster_labels == label, :] = dH_[label, :]
    return H, dH


def create_uncertainty(uncertainty, mask):
    """Creates the observational uncertainty matrix. We assume that
    uncertainty is a single value and we return a diagonal matrix back.
    We present this diagonal **ONLY** for pixels that have observations
    (i.e. not masked)."""
    good_obs = mask.sum()
    R_mat = np.ones(good_obs)*uncertainty*uncertainty
    return sp.dia_matrix((R_mat, 0), shape=(R_mat.shape[0], R_mat.shape[0]))


def create_linear_observation_operator(obs_op, n_params, metadata,
                                       mask, state_mask, 
                                       x_forecast, band=None):
    """A simple **identity** observation opeartor. It is expected that you
    subclass and redefine things...."""
    good_obs = mask.sum()  # size of H_matrix
    H_matrix = sp.dia_matrix(np.eye(good_obs))
    return H_matrix


def create_nonlinear_observation_operator(n_params, emulator, metadata,
                                          mask, state_mask,  x_forecast, band):
    """Using an emulator of the nonlinear model around `x_forecast`.
    This case is quite special, as I'm focusing on a BHR SAIL
    version (or the JRC TIP), which have spectral parameters
    (e.g. leaf single scattering albedo in two bands, etc.). This
    is achieved by using the `state_mapper` to select which bits
    of the state vector (and model Jacobian) are used."""
    LOG.info("Creating the ObsOp for band %d" % band)
    n_times = x_forecast.shape[0] / n_params
    good_obs = mask.sum()

    H_matrix = sp.lil_matrix((n_times, n_params * n_times),
                             dtype=np.float32)

    H0 = np.zeros(n_times, dtype=np.float32)

    # So the model has spectral components.
    if band == 0:
        # ssa, asym, TLAI, rsoil
        state_mapper = np.array([0, 1, 6, 2])
    elif band == 1:
        # ssa, asym, TLAI, rsoil
        state_mapper = np.array([3, 4, 6, 5])

    # This loop can be JIT'ed
    x0 = np.zeros((n_times, 4))
    for i in xrange(n_times):
        if mask[state_mask].flatten()[i]:
            x0[i, :] = x_forecast[(n_params * i) + state_mapper]
    LOG.info("Running emulators")
    # Calls the run_emulator method that only does different vectors
    # It might be here that we do some sort of clustering
    
    H0_, dH = run_emulator(emulator, x0[mask[state_mask]])
    
    LOG.info("Storing emulators in H matrix")
    # This loop can be JIT'ed too
    ########@jit
    #######def fast_assign(n_times, mask, state_mapper, n_params, dH,  H0_):
        
        #######H0 = np.zeros(n_times, dtype=np.float32)
        #######n_good = mask.sum()
        #######rows = np.zeros(n_good, dtype=np.int64)
        #######cols = np.zeros(n_good, dtype=np.int64)
        #######values = np.zeros(n_good, dtype=np.float32)
        #######n = 0
        #######m = 0
        #######for i in xrange(n_times):
            #######if mask.ravel()[i]:
                #######for jj,s in enumerate(state_mapper):
                    #######rows[m] = i
                    #######cols[m] = s + n_params*i
                    #######values[m] = dH[n, jj]
                    #######m += 1
                ########rows.append( i)
                ########cols.append(state_mapper + n_params * i)
                ########values.append(dH[n])
                ########H_matrix[i, state_mapper + n_params * i] = dH[n]
                #######H0[i] = H0_[n]
                #######n += 1
                #######return H0, rows, cols, values
    #######H0, rows, cols, values = fast_assign(n_times, mask, 
                    #######state_mapper, n_params, dH,  H0_)
    #######H_matrix = sp.coo_matrix ( (values, rows, cols), 
                              #######shape=[n_times, n_params * n_times]).tolil()
    n = 0
    for i in xrange(n_times):
        if mask[state_mask].flatten()[i]:
            H_matrix[i, state_mapper + n_params * i] = dH[n]
            H0[i] = H0_[n]
            n += 1
    LOG.info("\tDone!")

    return (H0, H_matrix.tocsr())



def locate_in_lut(lut, im):
    """This function locates a samples nearest neighbour in another dataset.
    We assume that `lut` is `[m, np]` and `im` is `[n, np]`, where `n >> m`
    and `np` is not too big. We will look for the location of the row of
    `lut` that is closest to each row in `im`.
    It returns `idx`, an array with an integer index to the first dimension 
    of lut."""
    assert ( lut.shape[1] == im.shape[1] )
    idx = np.linalg.norm ( lut[:, None, :] - im, axis=2).argmin(axis=0)
    return idx 


# This is a faster version for equally-sized blocks. 
#Currently, open PR on scipy's github
# (https://github.com/scipy/scipy/pull/5619)
def block_diag(mats, format=None, dtype=None):
    """
    Build a block diagonal sparse matrix from provided matrices.

    Parameters
    ----------
    mats : sequence of matrices
        Input matrices. Can be any combination of lists, numpy.array,
         numpy.matrix or sparse matrix ("csr', 'coo"...)
    format : str, optional
        The sparse format of the result (e.g. "csr").  If not given, the matrix
        is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output matrix.  If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    res : sparse matrix

    Notes
    -----
    Providing a sequence of equally shaped matrices
     will provide marginally faster results

    .. versionadded:: 0.18.0

    See Also
    --------
    bmat, diags, block_diag

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, block_diag
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5, 6], [7, 8]])
    >>> C = coo_matrix([[9, 10], [11,12]])
    >>> block_diag((A, B, C)).toarray()
    array([[ 1,  2,  0,  0,  0,  0],
           [ 3,  4,  0,  0,  0,  0],
           [ 0,  0,  5,  6,  0,  0],
           [ 0,  0,  7,  8,  0,  0],
           [ 0,  0,  0,  0,  9, 10],
           [ 0,  0,  0,  0, 11, 12]])
    """
    import scipy.sparse as sp
    import scipy.sparse.sputils as spu
    from scipy.sparse.sputils import upcast, get_index_dtype

    from scipy.sparse.csr import csr_matrix
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.bsr import bsr_matrix
    from scipy.sparse.coo import coo_matrix
    from scipy.sparse.dia import dia_matrix

    from scipy.sparse import issparse


    n = len(mats)
    mats_ = [None] * n
    for ia, a in enumerate(mats):
        if hasattr(a, 'shape'):
            mats_[ia] = a
        else:
            mats_[ia] = coo_matrix(a)

    if any(mat.shape != mats_[-1].shape for mat in mats_) or (
            any(issparse(mat) for mat in mats_)):
        data = []
        col = []
        row = []
        origin = np.array([0, 0], dtype=np.int)
        for mat in mats_:
            if issparse(mat):
                data.append(mat.data)
                row.append(mat.row + origin[0])
                col.append(mat.col + origin[1])

            else:
                data.append(mat.ravel())
                row_, col_ = np.indices(mat.shape)
                row.append(row_.ravel() + origin[0])
                col.append(col_.ravel() + origin[1])

            origin += mat.shape

        data = np.hstack(data)
        col = np.hstack(col)
        row = np.hstack(row)
        total_shape = origin
    else:
        shape = mats_[0].shape
        data = np.array(mats_, dtype).ravel()
        row_, col_ = np.indices(shape)
        row = (np.tile(row_.ravel(), n) +
               np.arange(n).repeat(shape[0] * shape[1]) * shape[0]).ravel()
        col = (np.tile(col_.ravel(), n) +
               np.arange(n).repeat(shape[0] * shape[1]) * shape[1]).ravel()
        total_shape = (shape[0] * n, shape[1] * n)

    return coo_matrix((data, (row, col)), shape=total_shape).asformat(format)


def spsolve2(a, b):
    a_lu = spl.splu(a.tocsc()) # LU decomposition for sparse a
    out = sp.lil_matrix((a.shape[1], b.shape[1]), dtype=np.float32)
    b_csc = b.tocsc()
    for j in xrange(b.shape[1]):
        bb = np.array(b_csc[j, :].todense()).squeeze()
        out[j,j] = a_lu.solve(bb)[j]
    return out.tocsr()


def matrix_squeeze (a_matrix, mask=None, n_params=1):
    """The matrix A is squeezed out of 0-filled submatrices, or rows/cols where
     a 1D mask indicates False. We return a squeezed version of the original
     matrix.
    Parameters
    ----------
    a_matrix: array
        An N,N array, or an N array. We assume that if no mask is given, we want
        to squeeze out all the zero locations
    mask: array
        An N boolean array, indicating where to squeeze the original array
    n_params: integer
        The number of parameters in the state vector. So for an identity o
        operator this will be 1, for the kernels 3, for TIP, 7, ...
    Returns
    -------
    The squeezed matrix.
    """
    if mask is None:
        # a needs to be a sparse matrix, otherwise find doesn't work!
        rows, columns, values = sp.find(a_matrix)
        # We can now squeeze easily using slicing of the original matrix
        a_matrix_squeezed = a_matrix[rows, :][:, columns]
    else:
        # Calculate the size of the output array from the non-zero mask elements
        if mask.ndim == 2:
            mask = mask.ravel()
        n = mask.sum()
        a_matrix_squeezed = sp.csr_matrix((n, n))
        m = np.array([], dtype=np.bool)
        # the next if statement is there to cope with problems where the size of
        # the state has more than one parameter
        if n_params > 1:
            # We need to stack the masks in this case
            for i in xrange(n_params):
                m = np.r_[m, mask]
        else:
            # We don't stack the masks, as there's only one parameter
            m = mask
        # This is different for vector and matrix
        if a_matrix.ndim == 2:
            # Just subset by mask location in rows/cols
            if a_matrix.getformat() == "dia":
                a_matrix_squeezed = a_matrix.tocsr()[m, :][:, m]
            else:
                a_matrix_squeezed = a_matrix[m, :][:, m]
        elif a_matrix.ndim == 1: # vector
            # Same, but just in one dimension
            a_matrix_squeezed = np.zeros(n_params*n)
            a_matrix_squeezed = a_matrix[m]
    return a_matrix_squeezed


def reconstruct_array(a_matrix, b_matrix, mask, n_params=1):
    """A function to fill in a squeezed array (a_matrix) with elements from a
    complete array (b_matrix). In effect, the elements of the b_matrix where the
    mask is True will be updated with the elements of a_matrix that correspond.
    The function works both on vectors and matrices, and they need to be
    ordered.
    Parameters
    -----------
    a_matrix: array
        The squeezed matrix with the updated elements
    b_matrix: array
        The full matrix that needs updating
    mask: array
        The location of the elements that need to be updated
    n_params: integer
        The number of parameters in the state
    Returns
    --------
    The updated `b_matrix`"""
    
    if mask.ndim > 1:
        mask = mask.ravel()
    n = mask.shape[0] # big dimension
    n_good = np.sum(mask)
    ilocs = mask.nonzero()[0]

    if a_matrix.ndim == 1:
        for i in xrange(n_params):
            b_matrix[ilocs + i*n] = a_matrix[(i*n_good):((i+1)*n_good)]
    elif a_matrix.ndim == 2:
        for i in xrange(n_params):
            ii = 0
            for j in xrange(n):
                if mask[j]:
                    b_matrix[j + i*n, i*n + ilocs] = a_matrix[ii,
                                                     (i*n_good):((i+1)*n_good)]
                    ii += 1
    return b_matrix


class OutputFile(object):
    """A netCDF4 class for saving output data. This class requires both the
    netCDF4 bindings, as well as the GDAL ones, as the class can take a GDAL
    projection to store it in the netCDF using CF-1.7. Lifted shamelessly from
    my eoldas code."""
    def __init__(self, fname, times=None, input_file=None,
                 basedate=dt.datetime(1970, 1, 1, 0, 0, 0),
                 x=None, y=None):
        """

        Parameters
        ----------
        fname : str
            A filename for the netCDF4 file.
        times : None
            Either None (default), or a list of datetime objects containing the
            dates of the data. The time axis of the data will be based on these.
            so you must prescribe the time before you save the data.
        input_file : str
            You can use a GDAL object (fully georeferenced) to copy the
            dimensions and projection.
        basedate : datetime
            The starting date, an arcane concept from netCDF.
        x : None or array
            A list of eastings if not passing an `input_file`.
        y : None or array
            A list of northings if not passing an `input_file`.
        """
        self.fname = fname
        self.basedate = basedate

        self.create_netcdf(times)

        if x is not None and y is not None:
            self.nc.createDimension('x', len(x))
            self.nc.createDimension('y', len(y))

            xo = self.nc.createVariable('x', 'f4', ('x'))
            xo.units = 'm'
            xo.standard_name = 'projection_x_coordinate'

            yo = self.nc.createVariable('y', 'f4', ('y'))
            yo.units = 'm'
            xo[:] = x
            yo[:] = y

        if input_file is not None:
            self._get_spatial_metadata(input_file)
            self.create_spatial_domain()


    def _get_spatial_metadata(self, geofile):
        """
        Gets (and sets!) the spatial metadata from a GDAL file
        Parameters
        ----------
        geofile : str
            The GDAL file. You might need to give a full path and what not...

        Returns
        -------
        Nuffink
        """
        g = gdal.Open(geofile)
        if g is None:
            raise IOError("File %s not readable by GDAL" % geofile)
        ny, nx = g.RasterYSize, g.RasterXSize
        geo_transform = g.GetGeoTransform()
        self.x = np.arange(nx) * geo_transform[1] + geo_transform[0]
        self.y = np.arange(ny) * geo_transform[5] + geo_transform[3]
        self.nx = nx
        self.ny = ny
        self.wkt = g.GetProjectionRef()


    def create_netcdf(self, times=None):
        """
        Creates our beloved netCDF file. Can also take an optional array of
        times.

        Parameters
        ----------
        times :

        Returns
        -------

        """
        self.nc = netCDF4.Dataset(self.fname, 'w', clobber=True)

        self.nc.createDimension('scalar', None)

        # create dimensions, variables and attributes:
        if times is None:
            self.nc.createDimension('time', None)
        else:
            # If we plan to append slices time must be unlimited
            self.nc.createDimension('time', None) #len(times))
        timeo = self.nc.createVariable('time', 'f4', ('time'))
        timeo.units = 'days since 1858-11-17 00:00:00'
        timeo.standard_name = 'time'
        timeo.calendar = "Gregorian"
        if times is not None:
            timeo[:] = netCDF4.date2num(times, units=timeo.units,
                                        calendar=timeo.calendar)

    def create_spatial_domain(self):
        self.nc.createDimension('x', self.nx)
        self.nc.createDimension('y', self.ny)

        xo = self.nc.createVariable('x', 'f4', ('x'))

        xo.units = 'm'
        xo.standard_name = 'projection_x_coordinate'

        yo = self.nc.createVariable('y', 'f4', ('y'))
        yo.units = 'm'
        yo.standard_name = 'projection_y_coordinate'
        self.nc.Conventions = 'CF-1.7'
        # create container variable for CRS: x/y WKT
        try:
            crso = self.nc.createVariable('crs', 'i4')
            crso.grid_mapping_name = "srs"
            crso.crs_wkt = self.wkt
        except AttributeError:
            print "Can't create a georeferenced netCDF file. Don't know why"
        xo[:] = self.x
        yo[:] = self.y


    def create_group(self, group):
        self.nc.createGroup(group)


    def create_variable(self,  varname, vardata,
                        units, long_name, std_name, vartype='f4', group=None):
        """
        Creates a variable in the file and add data to it. The idea being that
        this is where data gets stored.
        MISSING STUFF:
        * Chunking!
        Parameters
        ----------
        varname : str
            The variable name
        vardata : array
            The variable in a numpy array
        units : str
            The SI units (or more likely not)
        long_name : str
            The long variable name
        std_name : str
            The handy shorthand name
        vartype : str
            The variable type
        group : str
            The netCDF group where the variable goes
            
        Returns
        -------

        """
        self.create_empty_variable(varname, vardata.ndim, units, long_name,
                                   std_name, vartype=vartype, group=group,)
        if group is None:
            varo = self.nc.variables[varname]
        else:
            varo = self.nc.groups[group].variables[varname]
        if vardata.ndim == 3:
            varo[:, :, :] = vardata
        elif vardata.ndim == 2:
            varo[:,:] = vardata
        else:
            varo[:] = vardata

    def create_empty_variable(self, varname, ndim,
                              units, long_name, std_name, vartype='f4', group=None):
        """
        Creates a variable in the file without adding data. The idea being that
        this is where data gets stored.
        MISSING STUFF:
        * Chunking!
        Parameters
        ----------
        varname : str
            The variable name
        ndim : array
            The number of dimensions
        units : str
            The SI units (or more likely not)
        long_name : str
            The long variable name
        std_name : str
            The handy shorthand name
        vartype : str
            The variable type
        group : str
            The netCDF group where the variable goes

        Returns
        -------

        """
        if ndim == 1:
            args = [varname, vartype, ('time'),]
            kwargs = {'zlib':True, 'chunksizes':[16], 'fill_value':-9999}
        elif ndim == 2:
            args = [varname, vartype, ('y', 'x')]
            kwargs = {'zlib':True, 'chunksizes':[12, 12], 'fill_value':-9999}
        elif ndim == 3:
            args = [varname, vartype, ('time','y', 'x')]
            kwargs = {'zlib':True, 'chunksizes':[16, 12, 12], 'fill_value':-9999}
        else:
            args = [varname, vartype, 'scalar']
            kwargs = {}

        if group is None:
            varo = self.nc.createVariable(*args, **kwargs)
        else:
            varo = self.nc.groups[group].createVariable(*args, **kwargs)
        if ndim in [2,3]:
            varo.grid_mapping = 'crs'

        varo.units = units
        varo.scale_factor = 1.00
        varo.add_offset = 0.00
        varo.long_name = long_name
        varo.standard_name = std_name
        varo.set_auto_maskandscale(False)
        # varo[:,...] = vardata

    def update_time(self, time, index=np.s_[:]):
        """
        
        :param time: array
                    The times to be added to the time variable.
        :param index: slice
                    The slice defines where in the variable the times go.
        :return: 
        """
        varo = self.nc.variables['time']
        varo[index] = time

    def update_variable(self, varname, vardata, group=None):
        """
        Appends data to a variable in the file.
        MISSING STUFF:
        * Assumes variable already created
        * Chunking!
        Parameters
        ----------
        group : str
            The netCDF group where the variable goes
        varname : str
            The variable name
        vardata : array
            The variable in a numpy array

        Returns
        -------

        """
        if group is None:
            try:
                varo = self.nc.variables[varname]
            except KeyError:
                print "Variable ['{}'] not in ncfile.".format(varname)
                raise
        else:
            try:
                varo = self.nc.groups[group].variables[varname]
            except KeyError:
                print "Group ['{}'] and/or variable ['{}'] not in ncfile.".format(group, varname)
                raise
        if varo.dimensions[0] != 'time':
            raise TypeError("Can only append to a variable with time dimension (dimensions {})".format(varo.dimensions))
        if varo.ndim == len(vardata.shape):
            if varo.shape[1:] == vardata.shape[1:]:
                varo[varo.shape[0]:(varo.shape[0]+vardata.shape[0])] = vardata
            else:
                raise ValueError(
                    "Dimensions of new data {} don't match existing variable in netCDF file {}".format(
                        vardata.shape[1:], varo.shape[1:]
                    ))
        elif len(vardata.shape) == varo.ndim-1:
            if varo.shape[1:] == vardata.shape:
                varo[varo.shape[0], ...] = vardata
            else:
                raise ValueError(
                    "Dimensions of new data {} don't match existing variable in netCDF file {}".format(
                        vardata.shape, varo.shape[1:]
                    ))
        else:
            raise ValueError("Dimensions of new data don't match existing data.")


    def __del__(self):
        self.nc.close()
