import numpy as np
import pandas as pd
from functions import output_all_wafer_maps

data = np.load('data/wafermap_train.npy', allow_pickle = True)

# Create a DataFrame from the numpy array
df = pd.DataFrame(data)

#create directories for each failure type
output_all_wafer_maps(df)