from loading import load_dataset
from selection import train_test_eval, select_and_eval, print_selected, print_metrics, metric_random, metric_chi2

print
print "Loading a test dataset"
train, validation, test = load_dataset('test_data')

print
print "With all features..."
acc, f1, auc = train_test_eval(train.data, train.target, test.data, test.target)
print_metrics(acc, f1, auc)

features = train.feature_names
kfeatures = 2
print
print "With Chi-squared..."
selected = select_and_eval(metric_chi2, kfeatures, train.data, train.target, test.data, test.target)
print_selected(features, selected)

print
print "With random..."
selected = select_and_eval(metric_random, kfeatures, train.data, train.target, test.data, test.target)
print_selected(features, selected)
