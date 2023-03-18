import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
import utility.plot_settings

# -----------------------------------------------------
# Import data
# -----------------------------------------------------

df = pd.read_csv("../data/titanic.csv")

df["Age"].plot()
