import numpy as np
import pandas as pd
from utils import scale_dataframe, impute_dataframe, one_hot_dataframe

# write submission
output = pd.DataFrame(pred, columns=['Predicted'])
#test.join(output)[['Id','Predicted']].to_csv('predictions.csv', index=False)
