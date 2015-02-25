from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection import chi2 as metric_chi2
import numpy as np

class FeatureSelectionPipeline(BaseEstimator, SelectorMixin):
    def __init__(self, filters):
        self.filters = filters
        self.n_features = None
        
    def fit(self, X, y=None):
        """Fit all the filters"""
        self.n_features = X.shape[1]
        
        for filter in self.filters:
            X = filter.fit(X, y).transform(X)
            
        return self
    
    def transform(self, X):
        for filter in self.filters:
            X = filter.transform(X)
        return X

    def _get_support_mask(self):
        # Create a dataset from 1 to nfeatures
        feature_idx = np.empty((1, self.n_features))
        feature_idx[0] = np.array(range(self.n_features))
        
        # Filter it using the feature selection pipeline
        remaining = self.transform(feature_idx)[0]
        
        # Mark which features remain
        mask = np.zeros(self.n_features, bool)
        for fidx in remaining:
            mask[fidx] = True
            
        return mask

def feature_selector(scorer, kfeatures, X, y):
    remove_constants = VarianceThreshold()
    select_k_best = SelectKBest(score_func=scorer, k=kfeatures)
    pipe = FeatureSelectionPipeline([
        remove_constants,
        select_k_best,
    ])
    
    return pipe.fit(X=X, y=y)

def get_selected_feature_indices(selector):
    return selector.get_support(indices=True)

def filter_features(selector, X):
    return selector.transform(X)

def metric_random(X, y):
    num_features = X.shape[1]
    num_data = X.shape[0]

    from random import random

    scores = [random() for i in xrange(num_features)]
    pvalues = [0 for i in xrange(num_features)]
    return scores, pvalues


def quick_eval(X, y):
    kf = KFold(len(X), n_folds=2)
    acc = 0
    denom = 0
    for train, test in kf:
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        acc += train_test_eval(X_train, y_train, X_test, y_test)
        denom += 1
    print "Average accuracy over %d folds: %.1f%%" % (denom, 100 * acc / denom)

def train_test_eval(X_train, y_train, X_test, y_test):
    from sklearn import linear_model
    clf = linear_model.LogisticRegression(C=1).fit(X_train, y_train)
    return clf.score(X_test, y_test)

def print_selected(names, selected):
    print "Features:"
    for i, fname in enumerate(names):
        if i in selected:
            print " * %s" % fname
        else:
            print "   %s" % fname


def select_and_eval(features, metric, kfeatures, X_train, y_train, X_test=None, y_test=None):
    selector = feature_selector(metric, kfeatures, X_train, y_train)
    selected = get_selected_feature_indices(selector)
    X_selected = filter_features(selector, X_train)
    
    print_selected(features, selected)
    if X_test is None:
        quick_eval(X_selected, y_train)
    else:
        X_test_selected = filter_features(selector, X_test)
        acc = train_test_eval(X_selected, y_train, X_test_selected, y_test)
        print "Accuracy: %.1f%%" % (100 * acc)