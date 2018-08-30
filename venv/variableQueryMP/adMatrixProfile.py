import numpy as np
from use.timeseries import TimeSeries

import use.similarity_measures as sm
from variableQueryMP.iterationData import IterationData
import line_profiler
from bisect import bisect_left

#@profile
def computeMP(tsQuery, tsTarget, m, skip_step, mean, sigma, sigmaplus):
    #timeseries1: Query TS, timeseries2: Target TS
    QTList = []
    LBList = []
    n1 = len(tsQuery.timeseries)
    n2 = len(tsTarget.timeseries)
    indexes = n1 - m + 1
    MP12 = [] #Matrix Profile
    DP_all = {} # Distance Profiles for All Index in the timeseries
    idx = 0
    if int(m/skip_step)==0:
        step = 1
    else:
        step = int(m / skip_step)
    for index in range(0, indexes, step):
        data = tsTarget.timeseries
        index2 = index + m
        query = tsQuery.timeseries[index:index2]
        # compute Distance Profile(DP)
        # if std(query)==0, then 'mass_v2' will return a NAN, ignore this Distance profile
        #Numpy will generate the result with datatype 'float64', where std(query) maybe equals to 'x*e-17', but not 0
        #
        if sigma[tsQuery.name][index]<= 0.0001 or index >= len(sigmaplus[tsQuery.name]):
            continue
        else:
            DP_all[idx], QT, LB = sm.mass_v3(data, query, mean[tsTarget.name], mean[tsQuery.name][index], sigma[tsTarget.name], sigma[tsQuery.name][index], sigmaplus[tsQuery.name][index])
            MP12.append(min(DP_all[idx]))
            idx += 1
            QTList.append(QT)
            LBList.append(LB)
    return DP_all, MP12, QTList, LBList

#@profile
# input:    QT(m), LB(m+1), new_m(m+1), the incremental step is 1 by default
#           mean(m+1), sigma(m+1),
#           meanplus(m+2), sigmaplus(m+2)
# output:   DP(m+1), MP(m+1), qt(m+1), lb(m+2)
def updateMP(TS_Query, TS_Target, QT, LB, new_m, skip_step, mean, sigma, meanplus, sigmaplus):
    DP = {}
    MP = []
    lb = []
    qt = []
    n = len(TS_Query.timeseries)
    i = 0
    indexes = n - new_m + 1 # indexes decrease 1 than that of length = m
    if int(new_m/skip_step)==0:
        step = 1
    else:
        step = int(new_m / skip_step)
    for index in range(0, indexes, step):
        data = TS_Target.timeseries
        query = TS_Query.timeseries[index:index+new_m]
        if sigma[TS_Query.name][index] <= 0.0001 or index >= len(sigmaplus[TS_Query.name]):
            continue
        else:
            dp, min_dist, lbcopy, qtcopy = updateMpPart(query, TS_Query.name, index, data, TS_Target.name, LB[i], QT[i], mean, sigma, meanplus, sigmaplus)
        DP[i] = dp
        MP.append(min_dist)
        lb.append(lbcopy)
        qt.append(qtcopy)
        i = i + 1
    return DP, MP, qt, lb

#@profile
# from the given data of length =l, to get the minimal distance and Distance Profile for length =l+1
# input:    QT(m)
#           Q(m+1), LB(m+1), mean(m+1), sigma(m+1)
#           meanplus(m+2), sigmaplus(m+2)
# output:   DPplus(m+1), min_dist(m+1), LBplus(m+2), QTplus(m+1)
#def updateMpPart(lb_list, QT, Q, T, meanQ, sigmaQ, meanTList, sigmaTList):
def updateMpPart(Q, Qname, Qindex, ts_target, Tname, LB, QT, mean, sigma, meanplus, sigmaplus):
    iterationData = IterationData()
    QTplus = {}
    #for query of length=l-1, the number of mappings decreases by 1
    nbr_distance = len(LB) -1
    L = len(Q)
    DPplus = {}
    dist_list = []
    min_dist = None
    LB_values = [lb[1] for lb in LB]
    LB_index = [lb[0] for lb in LB]
    for i, (rawIdx, lb) in enumerate(LB):
        # check if rawIdx exists in the original list
        T_subseq = ts_target[rawIdx : rawIdx+L]
        # T_subseq is at the end of the timeseries
        if len(Q)!= len(T_subseq):
            #print("flag!!!!!")
            continue
        else:
            if rawIdx not in QT.keys():
                QT[rawIdx]  = np.dot(Q[:-1], T_subseq[:-1])
            QTplus[rawIdx] = iterationData.updateQT(QT[rawIdx], Q, T_subseq)
        dist_exact = sm.compute1Dist(mean[Qname][Qindex], mean[Tname][rawIdx], sigma[Qname][Qindex], sigma[Tname][rawIdx], QTplus[rawIdx], L)
        index = locateDistIndex(dist_exact, LB_values)
        if index+i >= nbr_distance:
            DPplus.update({rawIdx: dist_exact})
            continue
        elif i >= index:
            DPplus.update({rawIdx: dist_exact})
            min_dist = dist_exact
            break
        else:
            # find the original index in timeseries: QT, mean, sigma
            rawIndexList = locateDistIndexList(i, index, LB_index)
            # have found an exact distance in LB profile
            for idx in rawIndexList:
                # the boundary index of mean and meanplus is different
                if idx >= len(mean[Tname]):
                    continue
                if idx not in QT.keys():
                    QTplus[idx] = np.dot(Q, T_subseq)
                else:
                    QTplus[idx] = iterationData.updateQT(QT[idx], Q, T_subseq)

                dist_exact = sm.compute1Dist(mean[Qname][Qindex], mean[Tname][idx], sigma[Qname][Qindex],
                                             sigma[Tname][idx], QTplus[idx], L)
                DPplus.update({idx: dist_exact})
                dist_list.append(dist_exact)
            min_dist = min(dist_list)
            break
    if min_dist is None:
        min_dist = min(DPplus.values())
        #print("QTplus' key is: ", QTplus.keys())
        # limite the number of element in LBplus, decrease 1
        LBplus = sm.computeLB(QTplus, L, mean[Qname][Qindex], mean[Tname], sigma[Qname][Qindex], sigma[Tname], sigmaplus[Qname][Qindex])
    else:
        # no need to sort LBplus here, its order will follow the previous one
        LBplus = sm.linearComputeLB(LB, sigma[Qname][Qindex], sigmaplus[Qname][Qindex])


    return DPplus, min_dist, LBplus, QTplus

#@profile
def locateDistIndex(dist_exact, LB_values):
    #lb_list: [(rawIndex, LB)]
    index = bisect_left(LB_values, dist_exact)
    return index

#@profile
def locateDistIndexList(i, index, LB_index):
    #lb_list: [(rawIndex, LB)]
    rawIndexList = LB_index[i:index]
    return rawIndexList