from loading import load_dataset
import selection as s

print
print "Loading a test dataset"
train, validation, test = load_dataset('test_data')

print
print "With all features..."
acc, f1, auc = s.train_test_eval(train.data, train.target, test.data, test.target)
s.print_metrics(acc, f1, auc)

features = train.feature_names
kfeatures = 2
print
print "With Chi-squared..."
selected = s.select_and_eval(s.metric_chi2, kfeatures, train.data, train.target, test.data, test.target)
s.print_selected(features, selected)

print
print "With random..."
selected = s.select_and_eval(s.metric_random, kfeatures, train.data, train.target, test.data, test.target)
s.print_selected(features, selected)

print
print "With infogain..."
selected = s.select_and_eval(s.metric_infogain, kfeatures, train.data, train.target, test.data, test.target)
s.print_selected(features, selected)