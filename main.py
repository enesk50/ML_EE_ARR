import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ecgdetectors import Detectors

if __name__ == '__main__':
    # Insert own directory
    df = pd.read_csv(r'C:\Users\NS_tu\Documents\School\Q1\EE4C12 Machine Learning for EE\Project\CE_ARR\Project_Data_EE4C12_CE_ARR\ECG_data\Beats_nsvfq_48p.csv', header=None)
