# -*- coding: utf-8 -*-
""" Small script that shows hot to do one hot encoding
    of categorical columns in a pandas DataFrame.

    See:
    http://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
    http://scikit-learn.org/dev/modules/generated/sklearn.feature_extraction.DictVectorizer.html
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.base import TransformerMixin


def one_hot_dataframe(train, test, cols, replace=True):
    """ Takes train and test dataframes and a list of columns that need to be encoded.
    Returns a 3-tuple comprising the one-hot encoded dataframes and the fitted vectorizor.
    Modified from https://gist.github.com/kljensen/5452382
    """
    vec = DictVectorizer()    
    vecTrain = pd.DataFrame(vec.fit_transform(train[cols].to_dict(orient='records')).toarray())
    vecTest = pd.DataFrame(vec.transform(test[cols].to_dict(orient='records')).toarray())
    vecTrain.columns = vec.get_feature_names()
    vecTest.columns = vec.get_feature_names()
    vecTrain.index = train.index
    vecTest.index = test.index
    if replace is True:
        train = train.drop(cols, axis=1)
        train = train.join(vecTrain)
        test = test.drop(cols, axis=1)
        test = test.join(vecTest)
    return (train, test, vec)


def scale_dataframe(train, test, cols, replace=True):
    """ Takes train and test dataframes and a list of columns that need to be scaled.
    Returns a 3-tuple comprising the scaled dataframes and the fitted scaler.
    """
    scaler = StandardScaler()
    scaledTrain = pd.DataFrame(data=scaler.fit_transform(train[cols]), columns=cols)
    scaledTest = pd.DataFrame(data=scaler.transform(test[cols]), columns=cols)
    scaledTrain.columns = scaler.get_feature_names()
    scaledTest.columns = scaler.get_feature_names()
    scaledTrain.index = train.index
    scaledTest.index = test.index
    if replace is True:
        train = train.drop(cols, axis=1)
        train = train.join(scaledTrain)
        test = test.drop(cols, axis=1)
        test = test.join(scaledTest)
    return (train, test, scaler)


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], 
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

def impute_dataframe(train, test):
    return (DataFrameImputer().fit_transform(train), DataFrameImputer().fit_transform(test))