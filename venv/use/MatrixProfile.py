import numpy as np
from use.timeseries import TimeSeries
import use.similarity_measures as sm

def computeMP(timeseries1, timeseries2, subseq_length):
    #timeseries1: Query TS, timeseries2: Target TS
    t1 = timeseries1
    t2 = timeseries2
    n1 = len(t1.timeseries)
    n2 = len(t2.timeseries)
    indexes = n2 - subseq_length + 1
    MP12 = [float('inf')]* indexes #Matrix Profile
    IP12 = [0]* indexes #Index Profile
    DP_all = {} # Distance Profiles for All Index in the timeseries

    if (t1.name == t2.name):
        # self-similarity join, avoid trivial match
        flag = "self_similarity"
    else:
        # non self-similarity join
        flag = "non_self"
    for index in range(0, indexes):
        data = t1.timeseries
        index2 = index + subseq_length
        query = t2.timeseries[index:index2]
        # compute Distance Profile(DP)
        #DP = mass_v2(data, query)
        # if std(query)==0, then 'mass_v2' will return a NAN, ignore this Distance profile
        if np.std(query) == 0:
            continue
        else:
            DP_all[index] = sm.mass_v2(data, query)
            MP12, IP12 = updateMP_IP(MP12, DP_all[index], IP12, index, flag, subseq_length)
    return DP_all, MP12, IP12

def updateMP_IP(MP, DP, IP, index, flag, subseq_length):
    if (flag =="self_similarity"):
        range1 = max(0, index - subseq_length / 2)
        range2 = min(index + subseq_length / 2, len(MP))
        for i in range(0, int(range1)):
            if (MP[i] > DP[i]):
                MP[i] = DP[i]
                IP[i] = index
        for i in range(int(range2), len(MP)):
            if (MP[i] > DP[i]):
                MP[i] = DP[i]
                IP[i] = index
        return MP, IP
    else:
        for i in range(0, len(MP)):
            if (MP[i] > DP[i]):
                MP[i] = DP[i]
                IP[i] = index
        return MP, IP