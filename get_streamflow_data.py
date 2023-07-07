"""Get full data for a USGS gauge
"""

import numpy as np
import pandas as pd

from pygeohydro import NWIS


### 0: Specifications ###
station = ['01434000']
dates = ('1900-01-01', '2023-07-06')

nwis = NWIS()
Q = nwis.get_streamflow(station_ids=station, dates=dates)
Q.index = pd.to_datetime(Q.index.date)
Q = Q.dropna()

Q.to_csv(f'usgs_{station[0]}_daily_cms.csv', sep=',')