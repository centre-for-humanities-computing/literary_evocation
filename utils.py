# UTILS for the concreteness/SA study

import os

import json
import sklearn
import pandas as pd
from importlib import reload
import numpy as np

#nltk.download('punkt')

# plot
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px



# for calculating the measures
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()


# stats
from scipy.stats import mannwhitneyu, permutation_test
from scipy import stats
from scipy.stats import norm
from scipy.stats import spearmanr
from krippendorff import alpha as krippendorff_alpha
import pingouin as pg
from sklearn.metrics import root_mean_squared_error


# Roberta
#!pip install sentencepiece
#!pip install protobuf
from transformers import pipeline

# VADER
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
