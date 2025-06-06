import pandas as pd
import numpy as np
from dataprep_lite.cleaning import KNNImputerWrapper

if __name__ == '__main__':
    df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})
    imputer = KNNImputerWrapper()
    print(imputer.fit_transform(df))
