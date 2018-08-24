import use.MatrixProfile as mp
import use.shapelet as sp
import use.similarity_measures
import variableQueryMP.adMatrixProfile as admp
from use.timeseries import *
from utils import *
import numpy as np
from matplotlib import pyplot as plt

def extract_shapelet_all_length(k, dataset_list, pruning_option):
    #'dataset_list': [dict{}, dict{}, ...]
    dataset = {k: v for ds in dataset_list for k, v in ds.items()}
    # length of shapelet is from 1 to min_ts-1 in dataset
    min_m = float('inf')
    shap_list = []
    # 'ts' is the object of TimeSeries
    min_m = Utils.min_length_dataset(dataset.values())
    # m: 1, 2, ..., min_m-1
    #print("Maximum length of shapelet is : " + str(min_m))
    min_length = int(0.1 * min_m)
    max_length = int(0.5 * min_m)
    for m in range(min_length, max_length):
        admp.computeIterateMP()
        #print("Extracting shapelet length: " + str(m))
        #number of shapelet in shap_list: k * nbr_class * (min_l-1)
        nbr_candidate = int((min_m - m)/(0.25*m))
        if 0 < nbr_candidate < k :
            shap_list.extend(extract_shapelet(nbr_candidate, dataset, m, pruning_option))
        elif nbr_candidate > 0:
            shap_list.extend(extract_shapelet(k, dataset, m, pruning_option))

    # pruning by 'shapelet.normal_distance'
    ## order 'shap_list' by 'shapelet.normal_distance', descending order
    '''shap_list = sorted(shap_list, key=lambda x: x.normal_distance, reverse=True)
    shap_list = shap_list[:k]'''

    grouped_shapelets = {}
    list_all_shapelets_pruned = []
    for shap in shap_list:
        if shap.class_shapelet in grouped_shapelets.keys():
            grouped_shapelets[shap.class_shapelet].append(shap)
        else:
            grouped_shapelets[shap.class_shapelet] = [shap]
    for keyShapelet, groupShapelet in grouped_shapelets.items():
        list_shapelet_group = list(groupShapelet)
        shap_list_sorted = sorted(list_shapelet_group, key=lambda shap: shap.normal_distance, reverse=True)
        list_all_shapelets_pruned += list_shapelet_group[:int(k)]
    return list_all_shapelets_pruned

