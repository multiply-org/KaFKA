#!/usr/bin/env python
import _pickle as cPickle
import datetime
import glob
import os
import sys
import logging

import numpy as np
import scipy.sparse as sp # Required for unc
import gdal
import osr

import xml.etree.ElementTree as ET
from collections import namedtuple


# Set up logging
Log = logging.getLogger(__name__+".Sentinel2_Observations")


def parse_xml(filename):
    """Parses the XML metadata file to extract view/incidence 
    angles. The file has grids and all sorts of stuff, but
    here we just average everything, and you get 
    1. SZA
    2. SAA 
    3. VZA
    4. VAA.
    """
    with open(filename, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()

        vza = []
        vaa = []
        for child in root:
            for x in child.findall("Tile_Angles"):
                for y in x.find("Mean_Sun_Angle"):
                    if y.tag == "ZENITH_ANGLE":
                        sza = float(y.text)
                    elif y.tag == "AZIMUTH_ANGLE":
                        saa = float(y.text)
                for s in x.find("Mean_Viewing_Incidence_Angle_List"):
                    for r in s:
                        if r.tag == "ZENITH_ANGLE":
                            vza.append(float(r.text))
                            
                        elif r.tag == "AZIMUTH_ANGLE":
                            vaa.append(float(r.text))
                            
    return sza, saa, np.mean(vza), np.mean(vaa)


def reproject_image(source_img, target_img, dstSRSs=None):
    """Reprojects/Warps an image to fit exactly another image.
    Additionally, you can set the destination SRS if you want
    to or if it isn't defined in the source image."""
    g = gdal.Open(target_img)
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterXSize, g.RasterYSize
    xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
    xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
    ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
    ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])
    xRes, yRes = abs(geo_t[1]), abs(geo_t[5])
    if dstSRSs is None:
        dstSRS = osr.SpatialReference()
        raster_wkt = g.GetProjection()
        dstSRS.ImportFromWkt(raster_wkt)
    else:
        dstSRS = dstSRSs
    g = gdal.Warp('', source_img, format='MEM',
                  outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                  dstSRS=dstSRS)
    if g is None:
        raise ValueError("Something failed with GDAL!")
    return g


S2MSIdata = namedtuple('S2MSIdata',
                     'observations uncertainty mask metadata emulator')

class Sentinel2Observations(object):
    def __init__(self, parent_folder, emulator_folder, state_mask,
                 input_bands=None):
        if not os.path.exists(parent_folder):
            raise IOError("S2 data folder doesn't exist")
        
        # Here is where you set the bands you are interested in
        if input_bands is None:
            self.band_map = ['02', '03', '04', '05', '06', '07',
                              '08', '8A', '12']
        else:
            self.band_map = input_bands
        self.parent = parent_folder
        self.emulator_folder = emulator_folder
        self.state_mask = state_mask
        self._find_granules(self.parent)

        emulators = glob.glob(os.path.join(self.emulator_folder, "*.pkl"))
        emulators.sort()
        self.emulator_files = emulators

    def define_output(self):
        g = gdal.Open(self.state_mask)
        proj = g.GetProjection()
        geoT = np.array(g.GetGeoTransform())
        #new_geoT = geoT*1.
        #new_geoT[0] = new_geoT[0] + self.ulx*new_geoT[1]
        #new_geoT[3] = new_geoT[3] + self.uly*new_geoT[5]
        return proj, geoT.tolist() #new_geoT.tolist()


    def _find_granules(self, parent_folder):
        """Finds granules. Currently does so by checking for
        Feng's AOT file."""
        self.dates = []
        self.date_data = {}
        for root, dirs, files in os.walk(parent_folder):
            for fich in files:
                if fich.find("aot.tif") >= 0:
                    try:
                        this_date = datetime.datetime(
                            *[int(i) for i in root.split("/")[-4:-1]])
                    except:
                        a = root 
                        a = a.split('/')[-2].split('_')[-1]
                        this_date = datetime.datetime(int(a[:4]), int(a[4:6]), int(a[6:8]))
                    
                    self.dates.append(this_date)
                    self.date_data[this_date] = root
        self.bands_per_observation = {}
        
        for the_date in self.dates:
            #self.bands_per_observation[the_date] = 5 # 10 bands
            #  Put number of bands you are using here
            self.bands_per_observation[the_date] = len(self.band_map)

    def _find_emulator(self, sza, saa, vza, vaa):
        raa = vaa - saa
        vzas = np.array([float(s.split("_")[-3]) 
                         for s in self.emulator_files])
        szas = np.array([float(s.split("_")[-2]) 
                         for s in self.emulator_files])
        raas = np.array([float(s.split("_")[-1].split(".")[0]) 
                         for s in self.emulator_files])        
        e1 = szas == szas[np.argmin(np.abs(szas - sza))]
        e2 = vzas == vzas[np.argmin(np.abs(vzas - vza))]
        e3 = raas == raas[np.argmin(np.abs(raas - raa))]
        iloc = np.where(e1*e2*e3)[0][0]
        return self.emulator_files[iloc]

    def check_mask(self, band=0):
        g = gdal.Open(self.state_mask)
        s_mask = g.ReadAsArray().astype(np.bool)
        to_remove = []
        for date, thefile in self.date_data.items():
            print(thefile)
            data = self.get_band_data(date, band)
            mask = data.mask
            if sum(mask[s_mask]) == 0:
                to_remove.append(date)
        for date in to_remove:
            del self.date_data[date]
            self.dates.remove(date)
        Log.info("remove {}".format(to_remove))
        Log.info("keep {}".format(self.dates))


    def get_band_data(self, timestep, band):
        
        current_folder = self.date_data[timestep]


        meta_file = os.path.join(current_folder, "metadata.xml")
        try:
            sza, saa, vza, vaa = parse_xml(meta_file)
        except FileNotFoundError:
            meta_file = os.path.join(current_folder, "../MTD_TL.xml")
            sza, saa, vza, vaa = parse_xml(meta_file)
        metadata = dict (zip(["sza", "saa", "vza", "vaa"],
                            [sza, saa, vza, vaa]))
        # This should be really using EmulatorEngine...
        emulator_file = self._find_emulator(sza, saa, vza, vaa)
        emulator = cPickle.load( open (emulator_file, 'rb'),
                                 encoding='latin1')

        # Read and reproject S2 surface reflectance
        the_band = self.band_map[band]
        
       
        original_s2_file = os.path.join ( current_folder, 
                                         "B{}_sur.tif".format(the_band))
                                         
        # tree structure is different for AWS and scihub sentinel downloads. Newer versions of
        # atmospheric correction use scihub. First try assuming sci hub format, if fails try
        # AWS format.
        try:
            #raise SystemError
            s2_file = glob.glob(original_s2_file.split('IMG_DATA')[0]+'IMG_DATA/*_B*.tif')[0].split('_B')[0]+'_B%s_sur.tif'%the_band
            g = reproject_image(s2_file, self.state_mask)
        except SystemError:
            g = reproject_image(original_s2_file, self.state_mask)
            
        rho_surface = g.ReadAsArray()
        mask = rho_surface > 0
        rho_surface = np.where(mask, rho_surface/10000., 0)
        # Read and reproject S2 angles
        
        # ae to make finding the right emulator more intuitive
        band_dictionary = {'02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7, '08': 8, '8A': 9, '09': 10, '11': 12, '12': 13}
        
        emulator_band_map = []
        for i in self.band_map:
            emulator_band_map.append(band_dictionary[i])
        # ae
        
        # emulator_band_map = [2, 3, 4, 8, 13]
        
                
        R_mat = rho_surface*0.05
        R_mat[np.logical_not(mask)] = 0.
        N = mask.ravel().shape[0]
        R_mat_sp = sp.lil_matrix((N, N))
        R_mat_sp.setdiag(1./(R_mat.ravel())**2)
        R_mat_sp = R_mat_sp.tocsr()

        s2_band = bytes("S2A_MSI_{:02d}".format(emulator_band_map[band]), 'latin1')

        s2data = S2MSIdata(rho_surface, R_mat_sp, mask, metadata, emulator[s2_band])
       
        return s2data

