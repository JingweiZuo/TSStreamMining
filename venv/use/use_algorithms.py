import use.MatrixProfile
import use.shapelet
import use.similarity_measures
from use.timeseries import *
from utils import *

def findShapelet(timeseries, dataset, m):
    mp = MatrixProfile()
    #Matrix Profile Dictionary "mp_dict" and Index Profile Dictionary "ip_dict"
    mp_dict_same = []
    ip_dict_same = {}
    mp_dict_differ = []
    ip_dict_differ = {}
    #Matrix Profiles between the timeseries and all other TS in dataset
    mp_all ={}
    #Distance Profiles between
        # 1. "_list": all index in source timeseries and target timeseries
        # 2. "_all": source timeseries and all target TS in dataset
    dp_list_all = {}

    ts = TimeSeries()
    dict_ts_ByClass = ts.groupByClass_timeseries(dataset)
    list_class = dict_ts_ByClass.keys()

    for ts in dataset.values():
        # mp_dict_same: [mp1, mp2, ...], Array[Array[]]
        # ip_dict_same: {ts_name1:ip1, ts_name1:ip2, ...}, map(ts.name:Array[])
        # mp_all: {ts_name1:mp1, ts_name2:mp2, ...}, map(ts.name:Array[])
        # dp_list_all: {ts.target_name1:{index1:dp1, index2:dp2, ...}, ts.target_name2:{...}, ...}, map( ts_target.name: map(index:Array[]) )
        if (ts.class_timeseries == ts.class_timeseries):
            dp_list, mp_sameClass, ip_sameClass = mp.computeMP(timeseries, ts, m)
            mp_dict_same.append(mp_sameClass)
            ip_dict_same.update( {ts.name:ip_sameClass} )
            mp_all.update( {ts.name:mp_sameClass})
            dp_list_all.update( {ts.name:dp_list} )
        else:
            mp_differClass, ip_differClass = mp.computeMP(timeseries, ts, m)
            mp_dict_differ.append(mp_differClass)
            ip_dict_differ.update( {ts.name:ip_differClass} )
            mp_all.update({ts.name: mp_sameClass})

    # compute the average distance for each side (under the same class, or the different class)
    dist_side1 = np.mean(mp_dict_same, axis = 0)
    dist_side2 = np.mean(mp_dict_differ, axis = 0)

    # compute the difference of distance for 2 sides
    dist_differ = np.substract(dist_side2, dist_side1)
    dist_threshold = np.divide(np.add(dist_side1, dist_side2),2)

    # retrun the Distance Profiles, Matrix Profiles, distance difference, array size keeps the same,
    # map(ts_target.name: map(index_source:Array[])), map(ts_target.name:Array[]), Array[], Array[], Array[]
    return dp_list_all, mp_all, dist_differ, dist_threshold, ip_dict_same

'''
    Pruning, select top-k shapelets
'''
def extract_shapelet(k, dataset, m, pruning_option):
    # then check if the shapelet is in the timeseries, note timeseries' name
    dist_differ_list = {}
    dist_threshold_list = {}
    ip_dict_same_list = {}
    mp_all = {}
    class_list = []
    shapelet_list = []
    for ts in dataset.values:
        c = ts.class_timeseries
        class_list.append(c)
        # 'dp_list_all': map{ ts_name_source1: map{ts_target.name: map{index_source:Array[]}} },
        # 'mp_all': map{ ts_name_source1: map{ts_name_target1:Array[], ...}, ts_name_source2: map{...}, ... }
        dp_list_all[ts], mp_all[ts], dist_differ, dist_threshold, ip_dict_same = findShapelet(ts, dataset)
        # Array of distance's difference for all timeseries in the dataset
        # dist_differ_list[c]: {ts_name_source1:dp1, ts_name_source2:dp2, ...}, map(String:Array[])
        dist_differ_list[c].update( {ts.name:dist_differ} )
        # dist_threshold_list[c]: {ts_name_source1:dist_threshold1, ts_name_source2:dist_threshold2, ...}, map(String:Array[])
        dist_threshold_list[c].update( {ts.name:dist_threshold} )
        # ip_dict_same_list[c]: {ts_name_source1:{ts_name1:ip1, ts_name2:ip2, ...}, ts_name_source2:{ts_name1:ip1, ts_name2:ip2, ...}, ...}, map( String : map( String:Array[] ) )
        ip_dict_same_list[c].update( {ts.name:ip_dict_same} )

    # for each class, select top-k shapelets, then find the matching indices for top-k shapelets
    # top-k aims at the shapelets of different class, or top-k shapelets of each class?
    ## Here, we take k shapelets for each class
    if (pruning_option == "top_k"):
        for c in class_list:
            ts_namelist = dist_differ_list[c].keys()
            # take the k first values as the initial values, then update them
            keys = range(0, k)
            # take top k shapelets for each class
            topk_distdiff[c] = dict.fromkeys(keys, 0)

            for ts in ts_namelist:
                ## distance difference profile of timeseries 'ts'
                dp = dist_differ_list[c][ts]
                #'idx' is the position of max difference of distance for 'ts'
                for idx, dd in enumerate(dp):
                    min_val = min(topk_distdiff[c].values())
                    if (dd > min_val):
                        topk_distdiff[c] = {key:val for key, val in topk_distdiff[c].items() if val!=min_val}
                        topk_distdiff[c].update( { ts+"_"+str(idx) : dd} )

            # create shapelets and put matching timeseries
            #topk_distdiff[c]: {ts_name_source+index1 : distdiff1, ts_name_source+index2 : distdiff2, ... }
            for key, val in topk_distdiff[c].items():
                key_val = key.split("_")
                ts_name_source = key_val[0]
                ts_index_source = int(key_val[1])

                shap = shapelet()
                shap.class_shapelet = c
                shap.differ_distance = val
                shap.normal_distance = val / m ** 0.5
                shap.subsequence = dataset[ts_name_source][ts_index_source:ts_index_source + m]
                shap.name = hash(shap.subsequence)

                # 'dist_threshold_list[c]': {ts_name_source1:dist_threshold1, ts_name_source2:dist_threshold2, ...}, map(String:Array[])
                dist_thd = dist_threshold_list[c][ts_name_source][ts_index_source]
                shap.dist_threshold = dist_thd
                # find the distance in all timesereis in dataset, and compare it with dist_threshold,
                # ip_dict_same_list[c]: {ts_name_source1:{ts_name1:ip1, ts_name2:ip2, ...}, ts_name_source2:{ts_name1:ip1, ts_name2:ip2, ...}, ...}, map( String : map( String:Array[] ) )
                ip_list_all = ip_dict_same_list[c][ts_name_source]
                for ts_name_target, ip in ip_list_all.items():
                    # if we know 'idx_target', we can find the matching position in target time series with the minimal distance
                    # however, we need to find all matching position in target time series, so here 'idx_target' is useless
                    # idx_target = ip[ts_index_source]

                    # 'mp_all': map{ ts_name_source1: map{ts_name_target1:Array[], ...}, ts_name_source2: map{...}, ... }
                    dist = mp_all[ts_name_source][ts_name_target][ts_index_source]
                    if (dist <= dist_thd ):
                        shap.matching_ts.append(ts_name_target)
                        # find the Distance Profile of idx_source -> ts_target
                        # 'dp_list_all': map{ ts_name_source1: map{ts_target.name: map{index_source:Array[]}} },
                        dp = dp_list_all[ts_name_source][ts_name_target]
                        for idx_d, d in enumerate(dp):
                            if (d <= dist_thd):
                                # if it's not NULL, append the value to the original one
                                if ts_name_target in shap.matching_indices.keys():
                                    shap.matching_indices[ts_name_target] = shap.matching_indices[ts_name_target].append(idx_d)
                                else:
                                    shap.matching_indices[ts_name_target] = [idx_d]
                shapelet_list.append(shap)
        # for each class, we've token k shapelets, so the final result contains k * nbr(class) shapelets
        return shapelet_list

    elif (pruning_option=="over"):
        '''
            Pruning, use cover strategy and then select top-k shapelets
        '''
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
        return shapelet_list

def extract_shapelet_all_length(k, dataset, pruning_option):
    # length of shapelet is from 1 to min_ts-1 in dataset
    min_l = float('inf')
    shap_list = []
    for ts in dataset:
        size_ts = size(ts)
        if (size_ts < min_l):
            min_l = size_ts
    # l: 1, 2, ..., min_l-1
    for l in range(1, min_l):
        #number of shapelet in shap_list: k * nbr_class * (min_l-1)
        shap_list.extend(extract_shapelet(k, dataset, pruning_option))
    # pruning by 'shapelet.normal_distance'
    ## order 'shap_list' by 'shapelet.normal_distance'
    shap_list = sorted(shap_list, key=lambda x: x.normal_distance)
    shap_list = shap_list[:k]
    return shap_list