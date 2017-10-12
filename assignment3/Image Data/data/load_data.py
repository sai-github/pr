import numpy as np
import glob

data_forest_path = glob.glob("forest/*")
data_highway_path = glob.glob("highway/*")
data_tallbuilding_path = glob.glob("tallbuilding/*")


#load single file
forest_features_1 = np.loadtxt(data_forest_path[0])
