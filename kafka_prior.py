# maybe try adding smoother to the MODIS LAI profiles, so tweek the below function to add a smoother
import gdal
import numpy as np
import glob
import sys
sys.path.insert(0,'/home/acornelius/')
from startup import get_index, modisfname2date, change_time
from smoothn import smoothn

class kafka_prior():
    
    def __init__(self, dataframe, do_smooth = False, smooth_factor = 23):
        
        self.dataframe = dataframe
        self.do_smooth = do_smooth
        self.smooth_factor = smooth_factor
        
    def _loop_modis_files(self,in_template, in_pname, in_scale):
        
        h18_root = '/home/acornelius/MODIS/h18v03/%s/'%in_pname
        h17_root = '/home/acornelius/MODIS/h17v03/%s/'%in_pname
        
        for i in self.dataframe.values:
            
            lat, lon = i[[3, 4]]
            
            if lon > 0:
                flist = sorted(glob.glob(h18_root + '*.hdf'))
            else:
                flist = sorted(glob.glob(h17_root + '*.hdf'))

            ind = get_index([lat, lon], in_template % flist[0])

            prior_value = []
            prior_dates = []

            for j in flist:
                pix = gdal.Open(in_template % j).ReadAsArray(xoff=ind[1], yoff=ind[0],
                                                              xsize=1, ysize=1)[0][0] * in_scale

                prior_dates.append(modisfname2date(j))
                prior_value.append(pix)

            interp_daily = change_time(prior_dates, prior_value, '1d')
            
            if self.do_smooth == True:
                smooth = smoothn(y=np.array(interp_daily[1],dtype=float),
                                 s=self.smooth_factor)
                interp_daily[1] = smooth[0]

            self.ret.append(interp_daily)

        
    def get_modis_ndvi(self):
        
        template = 'HDF4_EOS:EOS_GRID:"%s":MODIS_Grid_16DAY_500m_VI:500m 16 days NDVI'
    
        scale_factor = 0.0001
        
        product_name = 'mod13a1'
        
        self.ret = []
        
        self._loop_modis_files(template, product_name, scale_factor)
        
    def get_modis_evi(self):
        
        template = 'HDF4_EOS:EOS_GRID:"%s":MODIS_Grid_16DAY_500m_VI:500m 16 days EVI'
        
        scale_factor = 0.0001
        
        product_name = 'mod13a1'
        
        self.ret = []
        
        self._loop_modis_files(template, product_name, scale_factor)
        
    def get_modis_lai(self):
        
        template = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_MOD15A2H:Lai_500m'
        
        scale_factor = 0.1
        
        product_name = 'mcd15a2h'
        
        self.ret = []
        
        self._loop_modis_files(template, product_name, scale_factor)
        
    def output(self):
        
        return self.ret