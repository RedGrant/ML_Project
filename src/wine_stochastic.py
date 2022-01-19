# FEUP | PDEEC | 2021/2022 | MACHINE LEARNING
# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt

# *********************************************************************************************************************#

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os