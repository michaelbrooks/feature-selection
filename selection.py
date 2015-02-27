import sys
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection import chi2 as metric_chi2
import numpy as np
import math


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


def entropy(x, y):
    """
    When the probability of x is 0, the entropy term for x
    is taken to be 0. Same for y.
    """
    if x > 0:
        p_x = x / float(x + y)
        e_x = p_x * math.log(p_x, 2)
    else:
        e_x = 0
    
    if y > 0:
        p_y = y / float(x + y)
        e_y = p_y * math.log(p_y, 2)
    else:
        e_y = 0
        
    return - e_x - e_y


def metric_infogain(X, y):
    """
    Compute the information gain for each feature
    according to Forman 2003.
    """
    POSITIVE = 1

    n_features = X.shape[1]
    n_cases = X.shape[0]

    # number of positive cases
    pos = 0
    # number of negative cases
    neg = 0

    for c in xrange(n_cases):
        label = y[c]
        if label == POSITIVE:
            pos += 1
        else:
            neg += 1
    
    baseline = entropy(pos, neg)

    scores = np.zeros(n_features)
    pvalues = np.zeros(n_features)

    for i in xrange(n_features):
        # number of positive cases containing the feature
        true_pos = 0
        # number of negative cases containing the feature
        false_pos = 0
        # number of positive cases not containing the feature
        false_neg = 0
        # number of negative cases not containing the feature
        true_neg = 0
        
        # Count cases for this feature
        for c in xrange(n_cases):
            featurevalue = X[c][i]
            label = y[c]
            if featurevalue > 0:
                if label == POSITIVE:
                    true_pos += 1
                else:
                    false_pos += 1
            else:
                if label == POSITIVE:
                    false_neg += 1
                else:
                    true_neg += 1
        
        pword = float(true_pos + false_pos) / n_cases
        scores[i] = baseline \
                    - pword * entropy(true_pos, false_pos) \
                    - (1 - pword) * entropy(false_neg, true_neg)

    return scores, pvalues


def quick_eval(X, y):
    kf = KFold(len(X), n_folds=2, shuffle=True)
    acc = 0
    f1 = 0
    pr_auc = 0
    denom = 0
    for train, test in kf:
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        _acc, _f1, _pr_auc = train_test_eval(X_train, y_train, X_test, y_test)
        acc += _acc
        f1 += _f1
        pr_auc += _pr_auc
        denom += 1
    print "Average over %d folds:"
    print_metrics(acc / denom, f1 / denom, pr_auc / denom)


def train_test_eval(X_train, y_train, X_test, y_test, model='lr'):
    """Get accuracy, F1, and PR AUC"""

    from sklearn.metrics import f1_score, average_precision_score, accuracy_score

    if model == 'lr':
        from sklearn import linear_model

        model = linear_model.LogisticRegression(C=1)
    elif model == 'svm':
        from sklearn import svm

        model = svm.LinearSVC(C=1)

    clf = model.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    y_score = clf.decision_function(X_test)

    f1 = f1_score(y_test, y_predicted)
    pr_auc = average_precision_score(y_test, y_score)
    acc = accuracy_score(y_test, y_predicted)
    return acc, f1, pr_auc


def print_selected(names, selected):
    print "Features:"
    for i, fname in enumerate(names):
        if i in selected:
            print " * %s" % fname
        else:
            print "   %s" % fname


def list_selected(names, selected, out=sys.stdout):
    for i, fname in enumerate(names):
        if i in selected:
            print >> out, fname


def print_metrics(train_acc, train_f1, train_pr_auc, test_acc=None, test_f1=None, test_pr_auc=None):
    if test_acc is None:
        print "    Acc: %0.1f%%" % (100 * train_acc)
        print "     F1: %.3f" % train_f1
        print "AUC(PR): %.3f" % train_pr_auc
    else:
        print "         Train   Test"
        print "    Acc: %0.1f%%  %0.1f%%" % (100 * train_acc, 100 * test_acc)
        print "     F1:  %.2f   %.2f" % (train_f1, test_f1)
        print "AUC(PR):  %.2f   %.2f" % (train_pr_auc, test_pr_auc)


def select_and_eval(metric, kfeatures, X_train, y_train, X_test=None, y_test=None):
    selector = feature_selector(metric, kfeatures, X_train, y_train)
    selected = get_selected_feature_indices(selector)
    X_selected = filter_features(selector, X_train)

    if X_test is None:
        quick_eval(X_selected, y_train)
    else:
        X_test_selected = filter_features(selector, X_test)
        train_acc, train_f1, train_pr_auc = train_test_eval(X_selected, y_train, X_selected, y_train)
        test_acc, test_f1, test_pr_auc = train_test_eval(X_selected, y_train, X_test_selected, y_test)
        print_metrics(train_acc, train_f1, train_pr_auc, test_acc, test_f1, test_pr_auc)

    return selected
