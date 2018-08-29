import numpy as np
from use.timeseries import TimeSeries

import use.similarity_measures as sm
from variableQueryMP.iterationData import IterationData
import line_profiler
from bisect import bisect_left

'''def computeIterateMP(TS_Query, TS_Target, L, newL, step):
    # Initalization
    DP = {}
    MP = {}
    n_Q = len(TS_Query)
    n_T = len(TS_Target)
    #Initialization: 'meanT/sigmaT' here is a list of list, same as DP[L]
    DP[L], MP[L], QT, meanQ, sigmaQ, meanT, sigmaT = mp.computeMP(TS_Query, TS_Target, L, step)
    LB = sm.computeLbGlobal(QT, )
    for l in range(L, newL):
        nbr_offset = int((n_Q - l) / step)
        for i in range(0, nbr_offset):
            index = i * step
            Q = TS_Query[index:index+l]
            #LB:{offset:lb_dict}, lb_dict: {rawIndex: LbDistance}
            lb_dict = LB[i]
            lb_list = [(idx, dist) for idx, dist in lb_dict.items()]
            lb_list = sorted(lb_list, key = lambda d:d[1])
            dp, min_dist, meanQ[i], sigmaQ[i], meanT[i], sigmaT[i], QT[i], flag = updateMpPart(lb_list, QT[l][i], Q, TS_Target, meanQ[i], sigmaQ[i], meanTList=meanT[i], sigmaTList=sigmaT[i])
            #dp is incomplete which doesn't contain all mappings, and just contain some smaller distances
            DP[l+1][i] = dp
            MP[l+1][i] = min_dist
            if flag:
                LB[i] = sm.linearComputeLB(LB[l+1][i], sigmaQ)
            else:
                LB[i] = sm.computeLB(QT, L, meanQ[L][i], meanT[L][i], sigmaQ[L][i], sigmaT[L][i], sigmaQ[L+1][i])
    return DP, MP'''

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
# newL here is L+1, the incremental step is 1 by default
def updateMP(TS_Query, TS_Target, QT, LB, new_m, skip_step, mean, sigma, meanplus, sigmaplus):

    ts_query = TS_Query.timeseries
    ts_target = TS_Target.timeseries
    n_Q = len(ts_query)
    if int(new_m/skip_step)==0:
        step = 1
    else:
        step = int(new_m / skip_step)
    nbr_offset = int((n_Q - new_m+1) / step)+1
    DP = {}
    MP = []
    lb = {}
    qt = {}

    IterationDataNewList = {}
    for i in range(0, nbr_offset):
        index = i * step
        Q = ts_query[index:index + new_m]
        '''if i >= len(QT):
            continue
        else:''''''for i in range(0, n_Q-new_m, step):
        #index = i * step'''
        #if round(iterationData.sigmaQ,4) == 0:
        if round(sigma[TS_Query.name][i], 4) == 0:
            continue
        else:
            dp, min_dist, lb[i], qt[i] = updateMpPart(Q, TS_Query.name, i, ts_target, TS_Target.name, LB[i], QT[i], mean, sigma, meanplus, sigmaplus)
        DP[i] = dp
        MP.append(min_dist)
    return DP, MP, qt, lb

#@profile
# from the given data of length =l, to get the minimal distance and Distance Profile for length =l+1
# Q(length = l+1), T; QT, meanQ, sigmaQ, meanTList, sigmaTList:(length = l)
#def updateMpPart(lb_list, QT, Q, T, meanQ, sigmaQ, meanTList, sigmaTList):
def updateMpPart(Q, Qname, Qindex, ts_target, Tname, LB, QT, mean, sigma, meanplus, sigmaplus):
    iterationData = IterationData()
    QTplus = {}
    #for query of length=l-1, the number of mappings decreases by 1
    nbr_distance = len(LB) -1
    '''print("nbr_distance is ", str(nbr_distance))
    print("nbr_QT is ", str(len(QT)))'''
    L = len(Q)
    DPplus = {}
    dist_list = []
    min_dist = None
    for i, (rawIdx, lb) in enumerate(LB):
        # check if rawIdx exists in the original list
        T_subseq = ts_target[rawIdx : rawIdx+L]
        #print("length Q is ", str(len(Q)), "length T_subseq is ", str(len(T_subseq)))
        if len(Q)!= len(T_subseq):
            #print("flag!!!!!")
            continue
        else:
            if rawIdx not in QT.keys():
                QT[rawIdx]  = np.dot(Q[:-1], T_subseq[:-1])
            QTplus[rawIdx] = iterationData.updateQT(QT[rawIdx], Q, T_subseq)
        dist_exact = sm.compute1Dist(meanplus[Qname][Qindex], meanplus[Tname][rawIdx], sigmaplus[Qname][Qindex], sigmaplus[Tname][rawIdx], QTplus[rawIdx], L)
        index = locateDistIndex(dist_exact, LB)
        if index+i >= nbr_distance:
            DPplus.update({rawIdx: dist_exact})
            continue
        elif i >= index:
            DPplus.update({rawIdx: dist_exact})
            min_dist = dist_exact
            break
        else:
            rawIndexList = locateDistIndexList(i, index, LB)

            # have found an exact distance in LB profile
            for idx in rawIndexList:
                # Two choses: - try to use the RAW index, or use the index in 'lb_list'
                #retrun to the original index in 'lb_list': QT, mean, sigma
                if idx not in QT.keys():
                    QTplus[idx] = np.dot(Q, T_subseq)
                else:
                    QTplus[idx] = iterationData.updateQT(QT[idx], Q, T_subseq)
                dist_exact = sm.compute1Dist(meanplus[Qname][Qindex], meanplus[Tname][idx], sigmaplus[Qname][Qindex],
                                             sigmaplus[Tname][idx], QTplus[idx], L)
                DPplus.update({idx: dist_exact})
                dist_list.append(dist_exact)
            min_dist = min(dist_list)
            break
    if min_dist is None:
        min_dist = min(DPplus.values())
        # limite the number of element in LBplus, decrease 1
        LBplus = sm.computeLB(QT, L, mean[Qname][Qindex], mean[Tname], sigma[Qname][Qindex], sigma[Tname], sigmaplus[Qname][Qindex])
    else:
        # no need to sort LBplus here, its order will follow the previous one
        LBplus = sm.linearComputeLB(LB, sigma[Qname][Qindex], sigmaplus[Qname][Qindex])


    return DPplus, min_dist, LBplus, QTplus

# to check the existence of dist in LB profile
'''def locateDistIndex(dist_exact, lb_list):
    #lb_list: [(rawIndex, LB)]
    for idx in range(1, len(lb_list)-1):
        low_value = lb_list.__getitem__(idx - 1)[1]
        high_value = lb_list.__getitem__(idx + 1)[1]
        if low_value <= dist_exact and high_value >= dist_exact:
            rawIndexList = []
            for i in range(0, idx):
                rawIndexList.append(lb_list.__getitem__(idx)[0])
            return idx, rawIndexList
    return None, None'''
#@profile
def locateDistIndex(dist_exact, lb_list):
    #lb_list: [(rawIndex, LB)]
    values = [lb[1] for lb in lb_list]
    index = bisect_left(values, dist_exact)
    return index

#@profile
def locateDistIndexList(i, index, lb_list):
    #lb_list: [(rawIndex, LB)]
    rawIndexList = [lb[0] for lb in lb_list[i:index]]
    return rawIndexList