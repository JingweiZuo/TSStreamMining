#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following line in the
entry_points section in setup.cfg:

    console_scripts =
     fibonacci = tasea.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""
from __future__ import division, print_function, absolute_import

import argparse
import sys
import logging
import time
import use.use_algorithms as use_algo
import use.old_use_algorithms as old_use_algo
import use.evaluation as ev
import gc
import csv
import os

from utils import Utils
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold

__author__ = "Jingwei ZUO"
__copyright__ = "Jingwei ZUO"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """
    Parse command lin-e parameters

    :param args: command line parameters as list of strings
    :return: command line parameters as :obj:`argparse.Namespace`
    """
    parser = argparse.ArgumentParser(
        description="Early classification on multivariate time series")

    parser.add_argument(
        '-d',
        '--data',
        dest="data_directory",
        help="Select the directory where the training csv files are saved",
        default=''
    )

    parser.add_argument(
        '-dm',
        '--distance',
        dest="distance_measure",
        help="Specify the distance measure. Three options are available: brute, mass, dtw.",
        default=''
    )
    parser.add_argument(
        '-c',
        '--cross',
        dest="cross",
        help="Do a cross validation",
        default='0'
    )
    parser.add_argument(
        '-sp',
        '--split',
        help="Split the dataset using scikit-learn."
             "Use this method instead of providing one data set for the learning and another for the testing",
        action='store_true'
    )
    parser.add_argument(
        '-k',
        '--topk',
        dest="top_k",
        help="Select the top-k shapelets.",
        default='20'
    )
    parser.add_argument(
        '-a',
        '--algo',
        dest="algo",
        help="Choose the old/new algorithm",
        default='use_old'
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    if args.distance_measure:
        measures = ['brute', 'mass_v2', 'dtw']
        if args.distance_measure in measures:
            distance_measure = args.distance_measure
        else:
            distance_measure = 'brute'
    else:
        distance_measure = 'brute'

    if not args.data_directory:
        print("No data directory is specified. Use the -d option, or -h for more help")
        sys.exit()

    top_k_value = int(args.top_k)
    #list_timeseries: Array[{ts_name:ts_value}]
    list_timeseries = Utils.generate_timeseries(args.data_directory)

    dataset = {k: v for ds in list_timeseries for k, v in ds.items()}
    y = [ts.class_timeseries for ts in dataset.values()]
    args.cross = int(args.cross)

    if args.cross:
        # sss = StratifiedShuffleSplit(n_splits=args.cross, test_size=0.25, random_state=0)
        skfold = StratifiedKFold(n_splits=args.cross)
        k = 1
        total_acc = 0.0
        total_app = 0.0
        dummy_list = list(range(len(list_timeseries)))
        gen = skfold.split(dummy_list, y)
        for train_index, test_index in gen:
            print("*******************************************************")
            print("Starting Fold Number", k, "out of", args.cross)
            list_ts_train = [list_timeseries[i] for i in train_index]
            list_ts_test = [list_timeseries[i] for i in test_index]

            start_time = time.time()
            print("Starting the USE algorithm...")
            if args.algo == "use_old":
                print("this is old algorithm")
                list_all_shapelets = old_use_algo.use_v4(list_ts_train, min_length=None, max_length=None,
                                                         pruning='top_k', k=top_k_value,
                                                         distance_measure=distance_measure, skip='True')
            elif args.algo == "use_new":
                list_all_shapelets = use_algo.extract_shapelet_all_length(top_k_value, list_ts_train, "top_k")



            print("USE algorithm complete")
            print("Time taken by the USE algorithm (minutes):", (time.time() - start_time) / 60)
            print("*******************************************************")
            print()
            print()

            print("Evaluating...")
            acc, sk_acc, sk_report, acc_maj, report_maj, app = ev.check_performance(list_ts_test,
                                                                                    list_all_shapelets,
                                                                                    distance_measure = distance_measure)
            # print("Accuracy:", acc)
            print("Applicability of Fold", k, ":", app, "%")
            total_app += app
            print("Total Accuracy:", acc, "%")
            total_acc += acc
            print("Classification Report:")
            print(sk_report)
            # if sk_acc:
            #     print("Accuracy of Fold (closest)", k, ":", sk_acc, "%")
            #     # print("Sk fscore:", sk_fscore)
            #     print("Classification Report (closets):")
            #     print(sk_report)
            #     total_acc += sk_acc
            # if acc_maj:
            #     print("Accuracy of Fold (majority voting)", k, ":", acc_maj, "%")
            #     # print("Sk fscore:", sk_fscore)
            #     print("Classification Report (majority voting):")
            #     print(report_maj)
            #     total_acc_maj += acc_maj
            print("Evaluation of fold", k, "complete")

            k += 1

        print("Total Applicability of the", args.cross, " Folds:", total_app / args.cross, "%")
        print("Total Accuracy of the", args.cross, " Folds:", total_acc / args.cross, "%")
        # if total_acc:
        #     print("Total Accuracy (closest) of the", args.cross, " Folds:", total_acc / args.cross, "%")
        # if total_acc_maj:
        #     print("Total Accuracy (majority voting) of the", args.cross, " Folds:", total_acc_maj / args.cross, "%")

        sys.exit()

    if args.split:
        list_ts_train, list_ts_test, y_train, y_test = train_test_split(list_timeseries, y, test_size=0.25, random_state=0)
        ###############################Save dataset to 'csv' file ###############################
        file_name = "dataset_test.csv"
        dirname = args.data_directory + "/csv_dataset/"
        ##clean the historical files
        files_list = [f for f in os.listdir(dirname) if f.lower().endswith('csv')]
        for file in files_list:
                    path = dirname + file
                    os.remove(path)
        path =  dirname + file_name
        with open(path, 'w') as f:
            writer = csv.writer(f, lineterminator='\n', delimiter=';',)
            for anObject in dataset.values():
                writer.writerow([anObject.name, anObject.class_timeseries])
        ###############################Save dataset to 'csv' file ###############################

        # Start of the USE algorithm
        # The USE algorithm
        start_time = time.time()
        if args.algo == "use_old":
            print("this is old use algorithm")
            list_all_shapelets = old_use_algo.use_v4(list_ts_train, min_length=None, max_length=None,
                                                     pruning='top-k', k=top_k_value,
                                                     distance_measure=distance_measure, skip=True)
        elif args.algo == "use_new":
            print("this is new use algorithm")
            list_all_shapelets = use_algo.extract_shapelet_all_length(top_k_value, list_ts_train, "top-k")

        print("USE algorithm complete")
        print("Time taken by the USE algorithm (minutes):", (time.time() - start_time) / 60)
        print("*******************************************************")
        #Utils.save(args.data_directory, list_all_shapelets, "shapelet")
        Utils.save(args.data_directory, list_all_shapelets, "csv")
        print()
        print()
        print("Evaluating...")
        acc, sk_acc, sk_report, acc_maj, report_maj, app = ev.check_performance(list_ts_test,
                                                                                list_all_shapelets,
                                                                                distance_measure=distance_measure)
        print("Applicability : ", app, "%")
        print("Accuracy acc: ", acc, "%", "Accuracy sk_acc: ", sk_acc, "%", "Accuracy acc_maj: ", acc_maj, "%")
        print("Classification Report:")
        print(sk_report)
def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
