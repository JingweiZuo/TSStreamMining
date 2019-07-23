'''SMAP: Shapelet on MAtrix Profile'''

import SMAP.MatrixProfile as mp
import similarity_measures as sm
import utils.utils as util
import numpy as np
import pandas as pd
import time, os
import re, math
from utils.old_Utils import old_Utils
import line_profiler

MP_data_file = None

class Shapelet(object):
    def __init__(self):
        self.id = id(self)
        self.name = ''
        self.subseq = None
        self.Class = ''
        self.differ_distance = 0.0
        self.normal_distance = 0.0
        self.dist_threshold = 0.0
        # [ts_target_name1, ts_target_name2, ...], Array[string]
        self.matching_ts = []
        # {ts_target_name:[idx1,idx2,...]}, dict{String:Array[]}
        self.matching_indices = {}

def computeMP(timeseries1, timeseries2, subseq_length, distance_measure):
    #timeseries1: Query TS, timeseries2: Target TS
    '''t1 = timeseries1
    t2 = timeseries2
    n1 = len(t1.timeseries)
    n2 = len(t2.timeseries)'''
    n1 = len(timeseries1)
    n2 = len(timeseries2)
    indexes = n1 - subseq_length + 1
    MP12 = [] #Matrix Profile
    #IP12 = [0] #Index Profile
    DP_all = {} # Distance Profiles for All Index in the timeseries
    idx = 0
    if int(subseq_length/4)==0:
        step = 1
    else:
        step = int(subseq_length / 4)
    for index in range(0, indexes, step):
        #data = t2.timeseries
        data = timeseries2
        index2 = index + subseq_length
        query = timeseries1[index:index2]
        # compute Distance Profile(DP)
        #DP = mass_v2(data, query)
        # if std(query)==0, then 'mass_v2' will return a NAN, ignore this Distance profile
        #Numpy will generate the result with datatype 'float64', where std(query) maybe equals to 'x*e-17', but not 0
        if round(np.std(query),4) == 0:
            continue
        else:
            DP_all[idx] = sm.calculate_distances(data, query, distance_measure)
            MP12.append(min(DP_all[idx]))
            idx += 1
    return DP_all, MP12

#@profile
def computeDistDiffer(timeseries, dataset, m, distance_measure):
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
    for ts in dataset.values():
        # mp_dict_same: [mp1, mp2, ...], Array[Array[]]
        # ip_dict_same: {ts_name1:ip1, ts_name1:ip2, ...}, dict(ts.name:Array[])
        # mp_all: {ts_target.name1:mp1, ts_target.name2:mp2, ...}, dict(ts_targe.name:Array[])
        # dp_all: {ts_target.name1:{index1:dp1, index2:dp2, ...}, ts_target.name2:{...}, ...}, dict(ts_target.name: dict(index:Array[]) )

        if timeseries.name != ts.name: #check the self-similarity
            if (timeseries.class_timeseries == ts.class_timeseries):
                #dp_list: {index:DP}
                #dp_list, mp_sameClass= computeMP(timeseries, ts, m, distance_measure)
                mp_sameClass= mp.computeMP(timeseries, ts, m, distance_measure)
                mp_dict_same.append(mp_sameClass)
                mp_all.update( {ts.name:mp_sameClass})
                #dp_all.update( {ts.name:dp_list} )
            else:
                #dp_list, mp_differClass= computeMP(timeseries, ts, m, distance_measure)
                mp_differClass = mp.computeMP(timeseries, ts, m, distance_measure)
                mp_dict_differ.append(mp_differClass)
                mp_all.update({ts.name: mp_differClass})
                #dp_all.update({ts.name: dp_list})

    '''START: Export the Intermediate Results of MP computing to csv file, for the usage of Shapelet Evaluation test'''
    global MP_data_file
    dataset_folder = '/'.join(MP_data_file.split('/')[:-1])
    files_list = [f for f in os.listdir(dataset_folder) if f.endswith('MPs.csv')]
    if files_list:
        df_MP = pd.read_csv(dataset_folder + '/' + files_list[0])
    else:
        df_MP = pd.DataFrame([[0, 0, 0, 0]],
                               columns=['ts_id', 'm', 'mp_dict_same', 'mp_dict_differ'])
    data = [timeseries.id, m, mp_dict_same, mp_dict_differ]
    df_MPdata = pd.DataFrame([data],
                           columns=['ts_id', 'm', 'mp_dict_same', 'mp_dict_differ'])
    df_MP = df_MP.append(df_MPdata)
    df_MP.to_csv(dataset_folder + "/IntermediateResultMPs.csv", index=False)
    '''END: Export the Intermediate Results of MP computing to csv file, for the usage of Shapelet Evaluation test'''

    # compute the average distance for each side (under the same class, or the different class)
    dist_side1 = np.mean(mp_dict_same, axis = 0)
    dist_side2 = np.mean(mp_dict_differ, axis = 0)
    # compute the difference of distance for 2 sides
    dist_differ = np.subtract(dist_side2, dist_side1)
    dist_threshold = dist_side1
    #dist_threshold = InformationGain(mp_dict_same, mp_dict_differ)

    # retrun the Distance Profiles, Matrix Profiles, distance difference, distance threshold, array size keeps the same
    # dict(ts_target.name: dict(index_source:Array[])), dict(ts_target.name:Array[]), Array[], Array[]
    #return dp_all, mp_all, dist_differ, dist_threshold
    return mp_all, dist_differ, dist_threshold

'''To convert the MP data to readable format, which can be processed by the algo.'''
def clearning(MP_data):
    mp_unit = MP_data.replace('[','').replace('(','').replace(']','').replace(')','')
    MP_list = mp_unit.split("array")[1:]
    MP_list_new = []
    for mp in MP_list:
        str_mp = mp.replace("\n", '').replace(' ', '')
        str_mp = [float(num) for num in re.split(',', str_mp) if num]
        MP_list_new.append(str_mp)
    return MP_list_new

def computeDistDifferFromFile(ts_id, m):
    global MP_data_file
    mp_all = {}
    dataset_folder = '/'.join(MP_data_file.split('/')[:-1])
    MPfile = dataset_folder + "/IntermediateResultMPs.csv"
    dfMP = pd.read_csv(MPfile)
    #filter the instance from the dataframe
    MP_oneInstance = dfMP[dfMP['ts_id'] == ts_id][dfMP['m'] == m]
    mp_dict_same = MP_oneInstance['mp_dict_same'].tolist()[0]
    mp_dict_differ = MP_oneInstance['mp_dict_differ'].tolist()[0]
    mp_dict_same = clearning(mp_dict_same)
    mp_dict_differ = clearning(mp_dict_differ)
    dist_side1 = np.mean(mp_dict_same, axis=0)
    dist_side2 = np.mean(mp_dict_differ, axis=0)
    # compute the difference of distance for 2 sides
    dist_differ = np.subtract(dist_side2, dist_side1)
    #dist_threshold = dist_side1
    dist_threshold = InformationGain(mp_dict_same, mp_dict_differ)
    return mp_all, dist_differ, dist_threshold

def InformationGain(mp_dict_same, mp_dict_differ):
    #Use Information Gain to decide the Split Point
    mp_list = []
    mp_list.extend(mp_dict_same)
    mp_list.extend(mp_dict_differ)
    dist_threshold = []
    for idx in range(0, len(mp_dict_same[0])):
        #for each position in MP, we compute the Information Gain as the split point of candidate Shapelets
        DistSet = []
        DistSet_C = []
        DistSet_NonC = []
        gain = float('-inf')
        thresh = 0
        for elem in mp_list:
            DistSet.append(elem[idx])
        for elem in mp_dict_same:
            DistSet_C.append(elem[idx])
        for elem in mp_dict_differ:
            DistSet_NonC.append(elem[idx])
        DistSet = sorted(DistSet)

        def MP_entropy(Size_C, Size_NonC):
            size = Size_C + Size_NonC
            p_C = float(Size_C) / size
            p_NonC = float(Size_NonC) / size
            return -(p_C * np.log2(p_C)) - (p_NonC * np.log2(p_NonC))
        for two_distances in old_Utils.sliding_window(DistSet, 2):
            two_distances = np.array(two_distances)
            candidate_dist_threshold = two_distances.mean()
            d1 = {'C':0, 'NonC':0}
            d2 = {'C':0, 'NonC':0}

            for elem in DistSet_C:
                ##divide the timeseries list into 2 parts: d1, d2
                if elem < candidate_dist_threshold:
                    d1['C'] += 1
                else:
                    d2['C'] += 1
            for elem in DistSet_NonC:
                ##divide the timeseries list into 2 parts: d1, d2
                if elem < candidate_dist_threshold:
                    d1['NonC'] += 1
                else:
                    d2['NonC'] += 1
            if (d1['C'] + d1['NonC'] !=0) and (d2['C'] + d2['NonC'] !=0):
                entropy_before_split = MP_entropy(len(DistSet_C), len(DistSet_NonC))
                entropy_after_split = MP_entropy(d1['C'], d1['NonC']) + MP_entropy(d2['C'], d2['NonC'])
                candidate_gain = entropy_before_split - entropy_after_split
                if candidate_gain > gain:
                    gain = candidate_gain
                    thresh = candidate_dist_threshold
                elif math.isnan(entropy_after_split):
                    gain = entropy_before_split
                    thresh = candidate_dist_threshold

        dist_threshold.append(thresh)
    return np.array(dist_threshold)
'''
    Pruning, select top-k shapelets
'''
#@profile
def extract_shapelet(k, dataset, m, pruning_option, distance_measure):
    # then check if the shapelet is in the timeseries, note timeseries' name
    dist_differ_list = {}
    dist_threshold_list = {}
    dp_all = {}
    mp_all = {}
    ip_all = {}
    class_list = []
    shapelet_list = []
    plot_flag = True
    for ts in dataset.values():
        c = ts.class_timeseries
        class_list.append(c)
        # 'dp_all': dict{ ts_name_source1: dict{ts_target.name: dict{index_source:Array[]}} },
        # 'mp_all': dict{ ts_name_source1: dict{ts_name_target1:Array[], ...}, ts_name_source2: dict{...}, ... }
        # 'ip_all': dict{ ts_name_source1: dict{ts_name_target1:Array[], ...}, ts_name_source2: dict{...}, ... }
        #dp_all[ts.name], mp_all[ts.name], dist_differ, dist_threshold= computeDistDiffer(ts, dataset, m, plot_flag)
        #mp_all[ts.name], dist_differ, dist_threshold = computeDistDiffer(ts, dataset, m, distance_measure)
        mp_all[ts.name], dist_differ, dist_threshold = computeDistDifferFromFile(ts.id, m)

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

                shap = Shapelet()
                shap.Class = c
                shap.differ_distance = val
                shap.normal_distance = val / (m ** 0.5)
                # To adjust the index position considering the step value
                ajt_idx = ts_index_source
                if int(m / 4) != 0:
                    step = int(m / 4)
                    ajt_idx = ts_index_source * step
                shap.subseq = dataset[ts_name_source].timeseries[ajt_idx:ajt_idx + m]
                #hashing the raw data of subseq as shapelet name
                shap.name = hash(shap.subseq.tostring())

                # 'dist_threshold_list[c]': {ts_name_source1:dist_threshold1, ts_name_source2:dist_threshold2, ...}, dict(String:Array[])
                dist_thd = dist_threshold_list[c][ts_name_source][ts_index_source]
                shap.dist_threshold = dist_thd
                # find the distance in all timesereis in dataset, and compare it with dist_threshold,
                # 'ip_all': dict{ ts_name_source1: dict{ts_name_target1:Array[], ...}, ts_name_source2: dict{...}, ... }
                '''ip_source_all = ip_all[ts_name_source]
                for ts_name_target, ip in ip_source_all.items():'''

                '''for ts_name_target in dataset.keys():
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
                                        shap.matching_indices[ts_name_target] = [idx_d]'''
                shapelet_list.append(shap)
        # for each class, we've token k shapelets, so the final result contains k * nbr(class) shapelets
        return shapelet_list
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

#@profile
def extract_shapelet_all_length(k, dataset, pruning_option, m_list, distance_measure, MP_file):
    global MP_data_file
    MP_data_file = MP_file
    #'dataset': {ts_name:ts_obj}
    shap_list = []
    # 'ts' is the object of TimeSeries
    min_m = util.min_length_dataset(dataset.values())
    for m in m_list:
        print("Extracting shapelet length: " + str(m))
        start = time.time()
        #number of shapelet in shap_list: k * nbr_class * (min_l-1)
        nbr_candidate = int((min_m - m)/(0.25*m))
        if 0 < nbr_candidate < k :
            shap_list.extend(extract_shapelet(nbr_candidate, dataset, m, pruning_option, distance_measure))
        elif nbr_candidate >= k:
            shap_list.extend(extract_shapelet(k, dataset, m, pruning_option, distance_measure))
        print("time consumed: ", str(time.time() - start))
    # pruning by 'shapelet.normal_distance'
    ## order 'shap_list' by 'shapelet.normal_distance', descending order
    '''shap_list = sorted(shap_list, key=lambda x: x.normal_distance, reverse=True)
    shap_list = shap_list[:k]'''

    grouped_shapelets = {}
    list_all_shapelets_pruned = []
    for shap in shap_list:
        if shap.Class in grouped_shapelets.keys():
            grouped_shapelets[shap.Class].append(shap)
        else:
            grouped_shapelets[shap.Class] = [shap]
    for keyShapelet, groupShapelet in grouped_shapelets.items():
        list_shapelet_group = list(groupShapelet)
        shap_list_sorted = sorted(list_shapelet_group, key=lambda shap: shap.normal_distance, reverse=True)
        list_all_shapelets_pruned += shap_list_sorted[:int(k)]
    return list_all_shapelets_pruned
