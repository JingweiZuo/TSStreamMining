import use.MatrixProfile
import use.shapelet
import use.similarity_measure
from use.timeseries import *
from venv.utils import *

def findShapelet(timeseries, dataset, m):
    mp = MatrixProfile()
    #Matrix Profile Dictionary "mp_dict" and Index Profile Dictionary "ip_dict"
    mp_dict_same = []
    ip_dict_same = {}
    mp_dict_differ = []
    ip_dict_differ = {}
    ts = TimeSeries()
    dict_ts_ByClass = ts.groupByClass_timeseries(dataset)
    list_class = dict_ts_ByClass.keys()

    for ts in dataset:
        # mp_dict_same: [mp1, mp2, ...], Array[Array[]]
        # ip_dict_same: {ts1:ip1, ts2:ip2, ...}, map(ts.name:Array[])
        if (ts.class_timeseries == ts.class_timeseries):
            mp_sameClass, ip_sameClass = mp.computeMP(timeseries, ts, m)
            mp_dict_same.append(mp_sameClass)
            ip_dict_same.extend( {ts.name:ip_sameClass} )
        else:
            mp_differClass, ip_differClass = mp.computeMP(timeseries, ts, m)
            mp_dict_differ.append(mp_differClass)
            ip_dict_differ.extend( {ts.name:ip_differClass} )
    # compute the average distance for each side (under the same class, or the different class)
    dist_side1 = np.mean(mp_dict_same, axis = 0)
    dist_side2 = np.mean(mp_dict_differ, axis = 0)
    # compute the difference of distance for 2 sides
    dist_differ = np.substract(dist_side2, dist_side1)
    dist_threshold = np.divide(np.add(dist_side1, dist_side2),2)
    # retrun the difference, array size keeps the same, Array[],Array[],Array[]
    return dist_differ, dist_threshold, ip_dict_same

def extract_shapelet(k, dataset, m):
    dist_differ_list = {}
    dist_threshold_list = {}
    ip_dict_same_list = {}
    class_list = []
    for ts in dataset:
        c = ts.class_timeseries
        class_list.append(c)
        dist_differ, dist_threshold, ip_dict_same = findShapelet(ts, dataset)
        # Array of distance's difference for all timeseries in the dataset
        # dist_differ_list[c]: {ts1.name:dp1, ts2.name:dp2, ...}, map(String:Array[])
        dist_differ_list[c].update( {ts.name:dist_differ} )
        # dist_threshold_list[c]: {ts1.name:dist_threshold1, ts2.name:dist_threshold2, ...}, map(String:Array[])
        dist_threshold_list[c].update( {ts.name:dist_threshold} )
        # ip_dict_same_list[c]: {ts1.name:{ts1:ip1, ts2:ip2, ...}, ts2.name:{ts1:ip1, ts2:ip2, ...}, ...}, map( String : map( String:Array[] ) )
        ip_dict_same_list[c].update( {ts.name:ip_dict_same} )
    # find matching TimeSeries of the shapelet


    for c in class_list:
        ts_namelist = dist_differ_list[c].keys()
        # take the k first values as the initial values, then update them
        keys = range(0, k)
        # take top k shapelets for each class
        topk_distdiff[c] = dict.fromkeys(keys, 0)
        for ts in ts_namelist:
            ## distance difference profile of timeseries 'ts'
            dp = dist_differ_list[c][ts]
            for idx, dd in enumerate(dp):
                min_val = min(topk_distdiff[c].values())
                if (dd > min_val):
                    topk_distdiff[c] = {key:val for key, val in topk_distdiff[c].items() if val!=min_val}
                    topk_distdiff[c].update( { ts+"_"+str(idx) : dd} )
        # create shapelets and put matching timeseries
        #topk_distdiff[c]: {ts.name_index1 : distdiff1, ts.name_index2 : distdiff2, ... }
        for key, val in topk_distdiff[c].items():
            key_val = key.split("_")
            ts_name = key_val[0]
            ts_index = int(key_val[1])
            # find the distance in all timesereis in dataset, and compare it with dist_threshold,
            # index in target time series
            index = ip_dict_same_list[c][ts_name][ts_index]

            # then check if the shapelet is in the timeseries, note timeseries' name

    # pruning shapelets by repetitive features

