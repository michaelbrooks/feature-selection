from sklearn import datasets
from selection import quick_eval, select_and_eval, metric_random, metric_chi2

# load the iris datasets
dataset = datasets.load_iris()
print "With all features..."
quick_eval(dataset.data, dataset.target)
print

features = dataset.feature_names
print
print "With Chi-squared..."
select_and_eval(features, dataset.data, dataset.target, metric_chi2, 2)

print
print "With random..."
select_and_eval(features, dataset.data, dataset.target, metric_random, 2)