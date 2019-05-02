import numpy as np
import similarity_measures as sm
import line_profiler
#@profile
def computeMP(timeseries1, timeseries2, subseq_length):
    #timeseries1: Query TS, timeseries2: Target TS
    t1 = timeseries1
    t2 = timeseries2
    n1 = len(t1.timeseries)
    n2 = len(t2.timeseries)
    indexes = n1 - subseq_length + 1
    MP12 = [] #Matrix Profile
    #IP12 = [0] #Index Profile
    DP_all = {} # Distance Profiles for All Index in the timeseries
    idx = 0
    '''if int(subseq_length/4)==0:
        step = 1
    else:
        step = int(subseq_length / 4)'''
    step = 1
    for index in range(0, indexes, step):
        data = t2.timeseries
        index2 = index + subseq_length
        #query = t2.timeseries[index:index2]
        query = t1.timeseries[index:index2]
        # compute Distance Profile(DP)
        #DP = mass_v2(data, query)
        # if std(query)==0, then 'mass_v2' will return a NAN, ignore this Distance profile
        #Numpy will generate the result with datatype 'float64', where std(query) maybe equals to 'x*e-17', but not 0
        if round(np.std(query),4) == 0:
            continue
        else:
            DP_all[idx] = sm.mass_v2(data, query)
            #DP_all[idx] = sm.euclidean_distance_unequal_lengths(data, query)
            MP12.append(min(DP_all[idx]))
            idx += 1
    return DP_all, MP12

def computeDistDiffer(timeseries, dataset, m):
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

        #if ts.name != timeseries.name: check the self-similarity
        if (timeseries.class_timeseries == ts.class_timeseries):
            dp_list, mp_sameClass= computeMP(timeseries, ts, m)
            mp_dict_same.append(mp_sameClass)
            mp_all.update( {ts.name:mp_sameClass})
            dp_all.update( {ts.name:dp_list} )

        else:
            dp_list, mp_differClass= computeMP(timeseries, ts, m)
            mp_dict_differ.append(mp_differClass)
            mp_all.update({ts.name: mp_differClass})
            dp_all.update({ts.name: dp_list})

    # compute the average distance for each side (under the same class, or the different class)
    dist_side1 = np.mean(mp_dict_same, axis = 0)
    dist_side2 = np.mean(mp_dict_differ, axis = 0)
    # compute the difference of distance for 2 sides
    dist_differ = np.subtract(dist_side2, dist_side1)
    dist_threshold = dist_side1
    # retrun the Distance Profiles, Matrix Profiles, distance difference, distance threshold, array size keeps the same
    # dict(ts_target.name: dict(index_source:Array[])), dict(ts_target.name:Array[]), Array[], Array[]
    return dp_all, mp_all, dist_differ, dist_threshold, dist_side1, dist_side2