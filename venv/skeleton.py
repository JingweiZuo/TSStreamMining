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
import tasea.tasea_machinelearning.tasea_algorithms as ust
import tasea.tasea_machinelearning.evaluation as ev
from tasea.ust_thread.thread_plots import PlottingThread
import gc
import csv
import os

from venv.utils import Utils
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

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    if args.distance_measure:
        measures = ['brute', 'mass', 'dtw']
        if args.distance_measure in measures:
            distance_measure = args.distance_measure
        else:
            distance_measure = 'brute'
    else:
        distance_measure = 'brute'

    if args.info:
        print("This code is developed by " + __author__)
        sys.exit()

    if not args.data_directory:
        print("No data directory is specified. Use the -d option, or -h for more help")
        sys.exit()

    list_multivariate_timeseries = Utils.convert_csv_to_multivariate_timeseries(args.data_directory)


    y = [mts.class_timeseries for mts in list_multivariate_timeseries]

    args.cross = int(args.cross)

    if args.cross:
        # sss = StratifiedShuffleSplit(n_splits=args.cross, test_size=0.25, random_state=0)
        skfold = StratifiedKFold(n_splits=args.cross)
        k = 1
        total_earliness = 0.0
        total_acc = 0.0
        total_app = 0.0
        dummy_list = list(range(len(list_multivariate_timeseries)))
        gen = skfold.split(dummy_list, y)
        for train_index, test_index in gen:
            print("*******************************************************")
            print("Starting Fold Number", k, "out of", args.cross)
            list_multi_train = [list_multivariate_timeseries[i] for i in train_index]
            list_multi_test = [list_multivariate_timeseries[i] for i in test_index]

            start_time = time.time()
            print("Starting the USE algorithm...")
            list_all_shapelets = ust.use_v4(list_multi_train, distance_measure=distance_measure)
            print("USE algorithm complete")
            print("Time taken by the USE algorithm (minutes):", (time.time() - start_time) / 60)
            print("*******************************************************")
            print()
            print()

            start_time = time.time()
            print("Starting the SEE algorithm...")
            list_all_sequences = ust.see_v2(list_all_shapelets, list_multi_train)
            print("SEE algorithm complete")
            print("Time taken by the SEE algorithm (minutes):", (time.time() - start_time) / 60)
            print("*******************************************************")
            print()
            print()

            print("Evaluating...")
            acc, earliness, sk_acc, sk_report, acc_maj, report_maj, app = ev.check_performance(list_multi_test,
                                                                                               list_all_sequences,
                                                                                               distance_measure=
                                                                                               distance_measure)
            # print("Accuracy:", acc)
            print("Applicability of Fold", k, ":", app, "%")
            total_app += app
            print("Earliness of Fold", k, ":", earliness, "%")
            total_earliness += earliness
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
        print("Total Earliness of the", args.cross, " Folds:", total_earliness / args.cross, "%")
        print("Total Accuracy of the", args.cross, " Folds:", total_acc / args.cross, "%")
        # if total_acc:
        #     print("Total Accuracy (closest) of the", args.cross, " Folds:", total_acc / args.cross, "%")
        # if total_acc_maj:
        #     print("Total Accuracy (majority voting) of the", args.cross, " Folds:", total_acc_maj / args.cross, "%")

        sys.exit()

    if args.split:
        list_multivariate_timeseries, list_multi_test, y_train, y_test = train_test_split(list_multivariate_timeseries,
                                                                                          y, test_size=0.25,
                                                                                          random_state=0)
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
            for anObject in list_multivariate_timeseries:
                writer.writerow([anObject.name, anObject.class_timeseries])
        ###############################Save dataset to 'csv' file ###############################
        args.eval_directory = 'done'

    if not args.skip1 and not args.skip2:
        # Start of the UST algorithm
        # The USE algorithm
        start_time = time.time()
        print("Starting the USE algorithm...")
        list_all_shapelets = ust.use_v4(list_multivariate_timeseries, distance_measure=distance_measure)
        print("USE algorithm complete")

        print("Time taken by the USE algorithm (minutes):", (time.time() - start_time) / 60)
        print("*******************************************************")
        #Utils.save(args.data_directory, list_all_shapelets, "shapelet")
        Utils.save(args.data_directory, list_all_shapelets, "csv")
        print()
        print()
    else:
        print("Skipping the USE algorithm and loading the shapelets from a directory")
        list_all_shapelets = Utils.load(args.data_directory, "shapelet")
        print("Loading complete")
        print("*******************************************************")
        print()
        print()

    if args.view_shapelet:
        t = PlottingThread(list_multivariate_timeseries, list_all_shapelets=list_all_shapelets)
        t.start()
        t.join()
        return

    if not args.skip2:
        # The SEE algorithm
        start_time = time.time()
        print("Starting the SEE algorithm...")
        list_all_sequences = ust.see_v2(list_all_shapelets, list_multivariate_timeseries)
        print("SEE algorithm complete")
        print("Time taken by the SEE algorithm (minutes):", (time.time() - start_time) / 60)

        print("*******************************************************")
        Utils.save(args.data_directory, list_all_sequences, "sequence")
        print()
        print()
    else:
        print("Skipping the SEE algorithm and loading the sequences from a directory")
        list_all_sequences = Utils.load(args.data_directory, "sequence")
        print("Loading complete")
        print("*******************************************************")
        print()
        print()

    if args.json:
        Utils.save_json(args.data_directory, list_all_sequences)

    if args.view_rule:
        t2 = PlottingThread(list_multivariate_timeseries, list_all_sequences=list_all_sequences)
        t2.start()
        t2.join()
        return
    # The TAQE algorithm
    # print("Starting the TAQE algorithm...")
    # list_all_sequences = ust.taqe(list_all_sequences, list_multivariate_timeseries)
    # print("TAQE algorithm complete")
    # print("*******************************************************")
    # print()
    # print()

    if args.eval_directory:
        print("Evaluating...")
        if args.eval_directory == 'done':
            list_multivariate_timeseries = list_multi_test
        else:
            list_multivariate_timeseries = Utils.convert_csv_to_multivariate_timeseries(args.eval_directory)
        acc, earliness, sk_acc, sk_report, acc_maj, report_maj, app = ev.check_performance(list_multivariate_timeseries,
                                                                                           list_all_sequences,
                                                                                           distance_measure=
                                                                                           distance_measure)
        print("Total Applicability:", app, "%")
        print("Total Earliness:", earliness, "%")
        print("Total Accuracy:", acc, "%")
        print("Classification Report:")
        print(sk_report)
        # if sk_acc:
        #     print("Total Accuracy (closest):", sk_acc, "%")
        #     print("Classification Report (closest):")
        #     print(sk_report)
        #
        # if acc_maj:
        #     print("Total Accuracy (majority voting):", acc_maj, "%")
        #     print("Classification Report (majority voting):")
        #     print(report_maj)
        print("Evaluation complete")
    _logger.info("Exiting ...")


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
