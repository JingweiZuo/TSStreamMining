import use.MatrixProfile as mp
import use.shapelet as sp
import use.similarity_measures
import variableQueryMP.adMatrixProfile as admp
from use.timeseries import *
from utils import *
import numpy as np
from matplotlib import pyplot as plt

min_length = 0

'''Optimisation of USE'''
def computeDistDiffer(timeseries, dataset, m, plot_flag, IterateData = {}):
    # Matrix Profile Dictionary "mp_dict", Distance Difference Profile, and Index Profile Dictionary "ip_dict"
    #'dataset': {key1:val1, key2:val2, ...}
    mp_dict_same = []

    mp_dict_differ = []
    ip_all = {}
    #Matrix Profiles between the timeseries and all other TS in dataset
    mp_all ={}
    #Distance Profiles between
        # 1. "_list": all index in source timeseries and target timeseries
        # 2. "_all": source timeseries and all target TS in dataset
    dp_all = {}
    IterateDataNew = {}
    for ts in dataset.values():
        # mp_dict_same: [mp1, mp2, ...], Array[Array[]]
        # ip_dict_same: {ts_name1:ip1, ts_name1:ip2, ...}, dict(ts.name:Array[])
        # mp_all: {ts_target.name1:mp1, ts_target.name2:mp2, ...}, dict(ts_targe.name:Array[])
        # dp_all: {ts_target.name1:{index1:dp1, index2:dp2, ...}, ts_target.name2:{...}, ...}, dict(ts_target.name: dict(index:Array[]) )
        global min_length
        if m == min_length:
            dp_list, mp, iteratedata = admp.computeMP(timeseries, ts, m)
        else:
            dp_list, mp, iteratedata = admp.updateMP(timeseries, ts, m, IterateData[ts.name])
        #if ts.name != timeseries.name: check the self-similarity
        if (timeseries.class_timeseries == ts.class_timeseries):
            mp_dict_same.append(mp)

        else:
            mp_dict_differ.append(mp)
        mp_all.update({ts.name: mp})
        dp_all.update({ts.name: dp_list})
        IterateDataNew.update({ts.name: iteratedata})

    # compute the average distance for each side (under the same class, or the different class)
    dist_side1 = np.mean(mp_dict_same, axis = 0)
    dist_side2 = np.mean(mp_dict_differ, axis = 0)
    # compute the difference of distance for 2 sides
    dist_differ = np.subtract(dist_side2, dist_side1)
    # Plot the diagram
    '''if plot_flag:
        plt.title("class of the TimeSeries: " + "class1")
        plt.xlabel("index/offset")
        plt.ylabel("Distance measure")
        X = range(0, len(dist_differ))
        plt.plot(X, dist_side1, color='green', linewidth=0.3, label='Distance to TS in class1')
        plt.plot(X, dist_side2, color='blue', linewidth=0.3, label='Distance to TS in non-class1')
        plt.plot(X, dist_differ, color='red', linewidth=0.3, label='Distance difference')
        plt.legend()
        plt.show()'''
    #dist_threshold = np.divide(np.add(dist_side1, dist_side2),2)
    dist_threshold = dist_side1
    # retrun the Distance Profiles, Matrix Profiles, distance difference, distance threshold, array size keeps the same
    # dict(ts_target.name: dict(index_source:Array[])), dict(ts_target.name:Array[]), Array[], Array[]
    return dp_all, mp_all, dist_differ, dist_threshold, IterateDataNew

'''
    Pruning, select top-k shapelets
'''
def computeAllData(dataset, m, plot_flag, IterateDataList):
    dist_differ_list = {}
    dist_threshold_list = {}
    class_list = []
    dp_all_list = {}
    mp_all_list = {}
    IterateDataListNew = {}

    for ts in dataset.values():
        c = ts.class_timeseries
        class_list.append(c)
        # 'dp_all': dict{ ts_name_source1: dict{ts_target.name: dict{index_source:Array[]}} },
        # 'mp_all': dict{ ts_name_source1: dict{ts_name_target1:Array[], ...}, ts_name_source2: dict{...}, ... }
        # 'ip_all': dict{ ts_name_source1: dict{ts_name_target1:Array[], ...}, ts_name_source2: dict{...}, ... }
        global min_length
        if m == min_length:
            dp_all_list[ts.name], mp_all_list[ts.name], dist_differ, dist_threshold, IterateDataListNew[ts.name] = computeDistDiffer(ts, dataset, m, plot_flag)
        else:
            dp_all_list[ts.name], mp_all_list[ts.name], dist_differ, dist_threshold, IterateDataListNew[ts.name] = computeDistDiffer(ts, dataset, m, plot_flag, IterateDataList[ts.name])
        plot_flag = False
        # Array of distance's difference for all timeseries in the dataset
        # dist_differ_list[c]: {ts_name_source1:dp1, ts_name_source2:dp2, ...}, dict(String:Array[])

        # dist_threshold_list[c]: {ts_name_source1:dist_threshold1, ts_name_source2:dist_threshold2, ...}, dict(String:Array[])
        if c in dist_differ_list.keys():
            dist_differ_list[c].update({ts.name:dist_differ})
            dist_threshold_list[c].update({ts.name:dist_threshold})
        else:
            dist_differ_list[c] = {ts.name:dist_differ}
            dist_threshold_list[c] = {ts.name: dist_threshold}
    return dist_differ_list, dist_threshold_list, dp_all_list, mp_all_list, class_list, IterateDataListNew

def extract_shapelet(k, dataset, m, pruning_option, IterateData):
    # then check if the shapelet is in the timeseries, note timeseries' name
    dist_differ_list = {}
    dist_threshold_list = {}
    dp_all = {}
    mp_all = {}
    ip_all = {}
    class_list = []
    shapelet_list = []
    plot_flag = True

    dist_differ_list, dist_threshold_list, dp_all, mp_all, class_list, IterateDataNew = computeAllData(dataset, m, plot_flag, IterateData)
    # for each class, select top-k shapelets, then find the matching indices for top-k shapelets
    # top-k aims at the shapelets of different class, or top-k shapelets of each class?
    ## Here, we take k shapelets for each class
    ### remove repetitive element in class_list
    class_list = list(set(class_list))
    if (pruning_option == "top-k"):
        for c in class_list:
            ts_namelist = dist_differ_list[c].keys()
            # take the k first values as the initial values, then update them
            keys = range(0, k)
            # take top k shapelets for each class
            topk_distdiff = dict.fromkeys(keys, float('-inf'))

            for ts in ts_namelist:
                ## distance difference profile of source timeseries 'ts'
                dp = dist_differ_list[c][ts]
                #'idx' is the position of max difference of distance for 'ts'
                for idx, dd in enumerate(dp):
                    # if we find an element in 'topk_distdiff' which is smaller than 'dd',
                    # then remove it and add 'dd' into 'topk', then break
                    min_topk = min(topk_distdiff.values())
                    for idx_topk, dd_topk in topk_distdiff.items():
                        if dd_topk == min_topk and dd_topk < dd:
                            topk_distdiff.pop(idx_topk)
                            key_composed = str(ts) + "_" + str(idx)
                            topk_distdiff.update({key_composed: dd})
                            break

            # create shapelets and put matching timeseries
            #topk_distdiff: {ts_name_source+index1 : distdiff1, ts_name_source+index2 : distdiff2, ... }
            for key, val in topk_distdiff.items():
                #print("key ", key, "value", val)
                key_val = key.split("_")
                ts_name_source = int(key_val[0])
                #the position of the shapelet in the source timeseries
                ts_index_source = int(key_val[1])

                shap = sp.Shapelet()
                shap.class_shapelet = c
                shap.differ_distance = val
                shap.normal_distance = val / (m ** 0.5)
                shap.subsequence = dataset[ts_name_source].timeseries[ts_index_source:ts_index_source + m]
                #hashing the raw data of subsequence as shapelet name
                shap.name = hash(shap.subsequence.tostring())

                # 'dist_threshold_list[c]': {ts_name_source1:dist_threshold1, ts_name_source2:dist_threshold2, ...}, dict(String:Array[])
                dist_thd = dist_threshold_list[c][ts_name_source][ts_index_source]
                shap.dist_threshold = dist_thd
                # find the distance in all timesereis in dataset, and compare it with dist_threshold,
                # 'ip_all': dict{ ts_name_source1: dict{ts_name_target1:Array[], ...}, ts_name_source2: dict{...}, ... }
                '''ip_source_all = ip_all[ts_name_source]
                for ts_name_target, ip in ip_source_all.items():'''
                for ts_name_target in dataset.keys():
                    # if we know 'idx_target', we can find the matching position in target time series with the minimal distance
                    # however, we need to find all matching position in target time series, so here 'idx_target' is useless
                    # idx_target = ip[ts_index_source]

                    # 'mp_all': dict{ ts_name_source1: dict{ts_name_target1:Array[], ...}, ts_name_source2: dict{...}, ... }
                    # we don't take self-similarity join, so we need to check source ts_name and target ts_name
                    if ts_name_source != ts_name_target:
                        dist = mp_all[ts_name_source][ts_name_target][ts_index_source]
                        if (dist <= dist_thd ):
                            shap.matching_ts.append(ts_name_target)
                            # find the Distance Profile of idx_source -> ts_target
                            # 'dp_all': dict{ ts_name_source1: dict{ts_target.name: dict{index_source:Array[]}} },
                            dp = dp_all[ts_name_source][ts_name_target][ts_index_source]
                            #dp:dict with less index then source ts, d:array with all index of source ts
                            for idx_d, d in enumerate(dp):
                                if (d <= dist_thd):
                                    # if it's not NULL, append the value to the original one
                                    if ts_name_target in shap.matching_indices.keys():
                                        shap.matching_indices[ts_name_target].append(idx_d)
                                    else:
                                        shap.matching_indices[ts_name_target] = [idx_d]
                shapelet_list.append(shap)
        # for each class, we've token k shapelets, so the final result contains k * nbr(class) shapelets
        return shapelet_list, IterateDataNew
    # pruning by checking if the list_shapelet covers the entire dataset
    '''elif (pruning_option=="cover"):
        #take top-K from every class, then check the coverage
        shapelet_list = []
        for aShapelet in list_shapelets:
            ##using the same training_data for all shapelets
            cover, remaining = aShapelet.cover(training_data)
            ##don't add repetitively the shapelet into the list
            if cover:
                shapelet_list.append(aShapelet)
            ##if remaining =0, that's to say the all timeseries in training data have been covered
            if not remaining:
                break
        return shapelet_list'''


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
    global min_length
    min_length = int(0.1 * min_m)
    max_length = int(0.5 * min_m)
    # for length = m, compute IterationDataList(QT, LB, mean, sigma)
    IterateData = {}

    for m in range(min_length, max_length):
        #print("Extracting shapelet length: " + str(m))
        #number of shapelet in shap_list: k * nbr_class * (min_l-1)
        nbr_candidate = int((min_m - m)/(0.25*m))
        if 0 < nbr_candidate < k :
            shapelets, IterateData = extract_shapelet(nbr_candidate, dataset, m, pruning_option, IterateData)
            shap_list.extend(shapelets)
        elif nbr_candidate > 0:
            shapelets, IterateData = extract_shapelet(k, dataset, m, pruning_option, IterateData)
            shap_list.extend(shapelets)

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

