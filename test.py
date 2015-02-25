from sklearn import datasets
from loading import load_dataset
from selection import quick_eval, train_test_eval, select_and_eval, metric_random, metric_chi2

# load the iris datasets
dataset = datasets.load_iris()
print "With all features..."
quick_eval(dataset.data, dataset.target)
print

features = dataset.feature_names
print
print "With Chi-squared..."
select_and_eval(features, metric_chi2, 2, dataset.data, dataset.target)

print
print "With random..."
select_and_eval(features, metric_random, 2, dataset.data, dataset.target)



print
print "Loading a test dataset"
train, validation, test = load_dataset('test_data')

print "Training data and targets:"
print train.data
print train.target

print "Test data and targets:"
print test.data
print test.target

print
print "With all features..."
acc = train_test_eval(train.data, train.target, test.data, test.target)
print "Accuracy: %.1f%%" % (100 * acc)

features = train.feature_names
kfeatures = 2
print
print "With Chi-squared..."
select_and_eval(features, metric_chi2, kfeatures, train.data, train.target, test.data, test.target)

print
print "With random..."
select_and_eval(features, metric_random, kfeatures, train.data, train.target, test.data, test.target)
