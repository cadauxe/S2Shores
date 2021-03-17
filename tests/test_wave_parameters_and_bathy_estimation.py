from wave_parameters_and_bathy_estimation import *
from shoresutils import *
import gdal
from scipy.io import loadmat

yaml_file = 'config/wave_bathy_inversion_config.yaml'
config = ConfigBathy('/work/OT/eolab/degoulr/bathyinversionvagues/config/wave_bathy_inversion_config.yaml')


data=loadmat('/work/LEGOS/shore/bathy_files/Synthetic_Rachid_1min_1Hz_1m.mat') # full data retrieved from MATLAB file
Im_MATLAB = np.reshape(data['M'],(-1,np.shape(data['Mt'])[1],np.shape(data['Mt'])[0]))[:,400:480,400:480]

sequence_subtile = gdal.Open('test_data/sequence_image_cap_breton.tif').ReadAsArray()
Im = sequence_subtile[:,400:600,400:600]

Im_MATLAB = permute_axes(Im_MATLAB)

result = wave_parameters_and_bathy_estimation(Im_MATLAB)
print(result)
