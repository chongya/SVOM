import pandas as pd
import numpy as np
from scipy import interp
from scipy.io import savemat

csv = pd.read_csv('LAITimeSeries2016.csv')
raw = csv['ACF']
t = pd.to_datetime(csv['Date'],format='%Y%m%d')
doy0 = t.dt.dayofyear
doy = np.arange(121,247)

data = interp(doy,doy0,raw)
savemat('CI.mat',{'data':data})
