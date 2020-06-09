#Numpy Imports
import numpy as np
from numpy.random import seed  as numpyseed
numpyseed(3)

#Pandas Imports
import pandas as pd 
from pandas import read_csv


#Datetime imports
from datetime import datetime
from datetime import timedelta

#Sklearn Imports
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn import linear_model



#Other Imports
import pickle
import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from collections import Counter
from scipy.stats import skew
from scipy import stats
import seaborn as sns
sns.set_style('whitegrid')



######
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# #Keras Imports
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from keras import regularizers






