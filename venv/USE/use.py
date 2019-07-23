import USE.use_shapelet as sp
import utils.similarity_measures
from utils.timeseries import *
from utils.use_utils import old_Utils
import numpy as np

import itertools
import operator
import threading
import gc
import sys

def use_v4(list_timeseries_dict, min_length=None, max_length=None, pruning="cover", k=20, distance_measure='brute', skip=False):
    # USE with psutil support
    # 'list_timeseries_dict': [dict{}, dict{}, ...]
    # 'list_timeseries': {ts_name : ts}
    list_timeseries = {k: v for ds in list_timeseries_dict for k, v in ds.items()}
    if not min_length:
        length = old_Utils.min_length_dataset(list_timeseries.values())
        min_length = int(length * 0.1)
        if length >= 40:
            max_length = int(length * 0.5)
        else:
            max_length = length - 1

    dict_timeseries_by_class = TimeSeries.groupByClass_timeseries(list_timeseries)
    list_all_shapelets_pruned = []
    print("Detecting " + str(len(dict_timeseries_by_class.keys())) + " classes")
    done = False
    list_remaining_cands = None
    while not done:
        list_done_shapelets, list_remaining_cands = uts_brute_force_v3(list(list_timeseries.values()), min_length, max_length,
                                                                           distance_measure=distance_measure, skip=skip,
                                                                           list_remaining_cands=
                                                                           list_remaining_cands)
        if not list_remaining_cands:
            done = True
        print("Starting the pruning procedure...")
        #grouped_shapelets = itertools.groupby(list_done_shapelets, lambda shapelet: shapelet.class_shapelet)
        grouped_shapelets = {}
        for shap in list_done_shapelets:
            if shap.class_shapelet in grouped_shapelets.keys():
                grouped_shapelets[shap.class_shapelet].append(shap)
            else:
                grouped_shapelets[shap.class_shapelet] = [shap]
        for keyShapelet, groupShapelet in grouped_shapelets.items():
            list_shapelet_group = list(groupShapelet)
            list_all_shapelets_pruned += pruning_shapelet(groupShapelet, algorithm=pruning, k=k,
                                                         training_data=list_timeseries)

        '''
        length = len(list_done_shapelets)
        # print("length of list_done_shapelets is ", len(list_done_shapelets))
        print("length of list_done_shapelets is ", len(list_done_shapelets))
        list_all_shapelets_pruned = pruning_shapelet(list_done_shapelets, algorithm=pruning, k = k, training_data = list_timeseries)
        # print("length of list_all_shapelets_pruned is ", len(list_all_shapelets_pruned))
        '''
        print("Pruning complete")
        print("*************************")
        list_done_shapelets = None
        gc.collect()

    #print("Length of list_all_shapelets_pruned is ", len(list_all_shapelets_pruned))
    print("Calculating the matching indices...")
    i = 0
    old_Utils.print_progress(i, length)
    for aShapelet in list_all_shapelets_pruned:
        aShapelet.build_matching_indices()
        i += 1
        old_Utils.print_progress(i, length)
    print("Calculation complete...")
    print("*************************")
    print()
    return list_all_shapelets_pruned

def uts_brute_force_v3(list_time_series, min_length, max_length, distance_measure='brute', skip=False,
                       list_remaining_cands=None):

    if not list_remaining_cands:
        print("Generating candidate shapelets for class: " + str(list_time_series[0].class_timeseries) + " ...")
        candidate_shapelets = sp.Shapelet.generate_candidates(list_time_series, min_length, max_length, skip=skip)
        print("Generation complete")
    else:
        print("Continue the learning on class: " + list_time_series[0].class_timeseries + " ...")
        candidate_shapelets = list_remaining_cands

    i = 0
    length = len(candidate_shapelets)

    print("Calculating the different features of the candidate shapelets...")
    old_Utils.print_progress(i, length)

    #tsclass = [ts.class_timeseries for ts in list_time_series]
    #print("tsclass is ", tsclass)
    for candidate in candidate_shapelets:##O(n*m2)
        candidate.gain, candidate.dist_threshold = candidate.check(list_time_series, distance_measure=distance_measure)
        #print("candidate.gain is ", str(candidate.gain), "candidate.dist_threshold is", candidate.dist_threshold)
        i += 1
        old_Utils.print_progress(i, length)
        good = old_Utils.check_memory()
        if not good:
            print()
            print("Memory usage is more that 90%")
            print("Forcing a prune phase, and then the learning will continue from where it stopped")
            return candidate_shapelets[:i], candidate_shapelets[i:]
    print("Calculation complete for dimension: ", list_time_series[0].dimension_name)
    print("*************************")
    # Return all the candidates
    return candidate_shapelets, None

def pruning_shapelet(list_shapelets, algorithm='cover', k=20, training_data=None):
    """
    :param list_shapelets: takes input as a list of shapelets
    :param algorithm: a parametrisation in order to add additional pruning algorithms
    :param k: if the algorithm is top_k we need to specify how much k we have to obtain from this pruning
    :param training_data: the training data set as a dict of timeseries by dimension
    :return: a list of pruned shapelets
    """
    if not list_shapelets:
        return []
    list_shapelets.sort(key=lambda x: x.gain, reverse=True)

    if algorithm == "top-k":
        return list_shapelets[:int(k)]

    if algorithm == "cover":
        print("the algorithm is cover")
        if not training_data:
            return []

        list_selected_shapelets = []
        for aShapelet in list_shapelets:
            ##using the same training_data for all shapelets
            cover, remaining = aShapelet.cover(training_data)
            ##don't add repetitively the shapelet into the list
            if cover:
                list_selected_shapelets.append(aShapelet)
            ##if remaining =0, that's to say the all timeseries in training data have been covered
            if not remaining:
                break
        return list_selected_shapelets

    else:
        return []