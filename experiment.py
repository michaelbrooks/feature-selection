#!/usr/bin/env python

import argparse
import loading
import selection
import csv
from path import path


def experiment(dataset_directory):
    train, validation, test = loading.load_dataset(dataset_directory)
    features = train.feature_names

    values_of_k = range(5, 50, 5)

    metrics = [
        selection.metric_chi2,
        selection.metric_random,
        selection.metric_infogain
    ]

    results = []

    for kfeatures in values_of_k:
        for metric in metrics:
            metric_name = metric.__name__
            print
            print "Testing k=%d, metric=%s" % (kfeatures, metric_name)

            selector = selection.feature_selector(metric, kfeatures, train.data, train.target)
            selected_indices = selection.get_selected_feature_indices(selector)

            train_data_selected = selection.filter_features(selector, train.data)
            test_data_selected = selection.filter_features(selector, test.data)

            train_acc, train_f1, train_pr_auc = selection.train_test_eval(train_data_selected, train.target,
                                                                          train_data_selected, train.target)
            test_acc, test_f1, test_pr_auc = selection.train_test_eval(train_data_selected, train.target,
                                                                       test_data_selected, test.target)
            selection.print_metrics(train_acc, train_f1, train_pr_auc, test_acc, test_f1, test_pr_auc)

            results.append(dict(
                kfeatures=kfeatures,
                metric=metric_name,
                train_accuracy=train_acc,
                train_f1=train_f1,
                train_pr_auc=train_pr_auc,
                test_accuracy=test_acc,
                test_f1=test_f1,
                test_pr_auc=test_pr_auc,
            ))

            output_name = dataset_directory / "features_%s_%d.csv" % (metric_name, kfeatures)
            with open(output_name, 'wb') as out:
                selection.list_selected(features, selected_indices, out=out)
                print "Features saved to %s" % output_name

    print
    print "Using all the features (Logistic Regression):"
    train_acc, train_f1, train_pr_auc = selection.train_test_eval(train.data, train.target, train.data, train.target)
    test_acc, test_f1, test_pr_auc = selection.train_test_eval(train.data, train.target, test.data, test.target)
    selection.print_metrics(train_acc, train_f1, train_pr_auc, test_acc, test_f1, test_pr_auc)

    results.append(dict(
        kfeatures=len(features),
        metric='logreg',
        train_accuracy=train_acc,
        train_f1=train_f1,
        train_pr_auc=train_pr_auc,
        test_accuracy=test_acc,
        test_f1=test_f1,
        test_pr_auc=test_pr_auc,
    ))

    print
    print "Using all the features (SVM):"
    train_acc, train_f1, train_pr_auc = selection.train_test_eval(train.data, train.target, train.data, train.target,
                                                                  model='svm')
    test_acc, test_f1, test_pr_auc = selection.train_test_eval(train.data, train.target, test.data, test.target,
                                                               model='svm')
    selection.print_metrics(train_acc, train_f1, train_pr_auc, test_acc, test_f1, test_pr_auc)

    results.append(dict(
        kfeatures=len(features),
        metric='svm',
        train_accuracy=train_acc,
        train_f1=train_f1,
        train_pr_auc=train_pr_auc,
        test_accuracy=test_acc,
        test_f1=test_f1,
        test_pr_auc=test_pr_auc,
    ))

    experiment_stats = dataset_directory / "feature_selection_experiment.csv"

    with open(experiment_stats, 'wb') as stats:
        writer = csv.DictWriter(stats, fieldnames=(
            'metric', 'kfeatures',
            'train_accuracy', 'train_f1', 'train_pr_auc',
            'test_accuracy', 'test_f1', 'test_pr_auc'))
        writer.writeheader()
        writer.writerows(results)

    print "Saved results in %s" % experiment_stats

def run():
    parser = argparse.ArgumentParser(description='Select features on a dataset using a bunch of algorithms.')
    parser.add_argument('dataset_directory', metavar='DIRECTORY', type=str,
                        help='a path to the dataset directory')
    parser.add_argument('--recursive', '-r', action='store_true', dest='recursive', default=False,
                        help='the given directory contains multiple datasets')
    args = parser.parse_args()

    dataset_directory = path(args.dataset_directory)
    if args.recursive:
        directory = path(dataset_directory)
        for dataset in directory.dirs():
            experiment(dataset)
    else:
        experiment(dataset_directory)


if __name__ == '__main__':
    run()
