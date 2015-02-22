from sklearn.cross_validation import KFold
from sklearn import datasets
from sklearn import svm
from sklearn.feature_selection import SelectKBest, chi2

def select_features(scorer, kfeatures, featurevectors, labels, **kwargs):
    selector = SelectKBest(score_func=scorer, k=kfeatures)
    selector = selector.fit(X=featurevectors, y=labels, **kwargs)
    return selector.get_support(indices=True), selector.transform(featurevectors)

def random(X, y):
    num_features = X.shape[1]
    num_data = X.shape[0]
    
    from random import random
    
    scores = [random() for i in xrange(num_features)]
    pvalues = [0 for i in xrange(num_features)]
    return scores, pvalues

def print_selected(names, selected):
    
    print "All features:"
    for fi in names:
        print fi
    print

    print "Selected features:"
    for fi in selected:
        print names[fi]

        
def quick_eval(X, y):
    kf = KFold(len(X), n_folds=2)
    acc = 0
    denom = 0
    for train, test in kf:
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        
        acc += clf.score(X_test, y_test)
        denom += 1
    print "Average accuracy over %d folds: %.1f%%" % (denom, 100 * acc / denom)

def select_and_eval(features, X, y, metric, kfeatures):
    best, bestX = select_features(metric, kfeatures, X, y)
    print_selected(features, best)
    quick_eval(bestX, y)

    
# load the iris datasets
dataset = datasets.load_iris()
print "With all features..."
quick_eval(dataset.data, dataset.target)
print

features = dataset.feature_names
print
print "With Chi-squared..."
select_and_eval(features, dataset.data, dataset.target, chi2, 2)

print
print "With random..."
select_and_eval(features, dataset.data, dataset.target, random, 2)