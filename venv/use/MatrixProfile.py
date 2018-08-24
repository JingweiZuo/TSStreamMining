import numpy as np
from use.timeseries import TimeSeries
import use.similarity_measures as sm

'''to complete the motification of step'''
def computeMP(timeseries1, timeseries2, subseq_length, step):
    #timeseries1: Query TS, timeseries2: Target TS
    t1 = timeseries1
    t2 = timeseries2
    n1 = len(t1.timeseries)
    n2 = len(t2.timeseries)
    indexes = n1 - subseq_length + 1
    step = int(subseq_length/4)
    MP12 = [] #Matrix Profile
    #IP12 = [0] #Index Profile
    DP_all = {} # Distance Profiles for All Index in the timeseries
    idx = 0
    if int(subseq_length/4)==0:
        step = 1
    else:
        step = int(subseq_length / 4)
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
            MP12.append(min(DP_all[idx]))
            idx += 1
    return DP_all, MP12