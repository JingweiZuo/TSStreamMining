import numpy as np
from use.timeseries import TimeSeries
import use.similarity_measures

class MatrixProfile(object):
    t1 = TimeSeries()
    t2 = TimeSeries()
    def computeMP(self, timeseries1, timeseries2, subseq_length):
        #timeseries1: Query TS, timeseries2: Target TS
        t1 = timeseries1
        t2 = timeseries2
        n1 = len(t1.timeseries)
        n2 = len(t2.timeseries)
        MP12 = [float('inf')] #Matrix Profile
        IP12 = [0] #Index Profile
        DP_all = {} # Distance Profiles for All Index in the timeseries
        indexes = n2-subseq_length+1

        if (t1.name == t2.name):
            # self-similarity join, avoid trivial match
            flag = "self_similarity"
        else:
            # non self-similarity join
            flag = "non_self"
        for index in range(0, indexes):
            data = t1.timeseries
            query = t2.timeseries[index:index + subseq_length]
            # compute Distance Profile(DP)
            #DP = mass_v2(data, query)
            DP_all[index] = mass_v2(data, query)
            MP12, IP12 = updataMP_IP(MP12, DP, IP12, index, flag)
        return DP_all, MP12, IP12

    def updateMP_IP(self, MP, DP, IP, index, flag):
        if (flag =="self_similarity"):
            range1 = max(0, index - subseq_length / 2)
            range2 = min(index + subseq_length / 2, len(MP))
            for i in range(0, range1) + range(range2, len(MP)) :
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