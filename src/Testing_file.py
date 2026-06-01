import pandas as pd
import numpy as np
from tabular import miss_case_cumsum
test_df = pd.DataFrame({
    "A": [1, np.nan, 3, np.nan, np.nan, 6, 7, np.nan, 9, 10,
          11, 12, np.nan, 14, 15, np.nan, 17, 18, 19, np.nan,
          21, 22, 23, np.nan, 25, 26, 27, 28, np.nan, 30],

    "B": [np.nan, 2, 3, np.nan, 5, 6, np.nan, 8, 9, np.nan,
          11, np.nan, 13, 14, 15, 16, np.nan, 18, 19, 20,
          np.nan, 22, 23, 24, 25, np.nan, 27, 28, 29, 30],

    "C": [1, 2, np.nan, np.nan, np.nan, 6, 7, 8, np.nan, 10,
          11, 12, 13, np.nan, 15, 16, 17, np.nan, 19, 20,
          21, 22, np.nan, 24, 25, 26, np.nan, 28, 29, 30],

    "D": [1, 2, 3, 4, np.nan, np.nan, 7, 8, 9, 10,
          np.nan, 12, 13, 14, 15, np.nan, 17, 18, np.nan, 20,
          21, np.nan, 23, 24, 25, 26, 27, np.nan, 29, 30],

    "E": [np.nan, 2, 3, 4, 5, np.nan, np.nan, 8, 9, 10,
          11, 12, np.nan, 14, 15, 16, 17, np.nan, 19, 20,
          21, 22, 23, 24, np.nan, 26, 27, 28, 29, np.nan],

    "F": [1, 2, np.nan, 4, 5, 6, np.nan, np.nan, 9, 10,
          11, np.nan, 13, 14, 15, 16, 17, 18, np.nan, 20,
          21, 22, np.nan, 24, 25, 26, 27, np.nan, 29, 30]
})

result = miss_case_cumsum(test_df)
print(result)
