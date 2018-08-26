import numpy as np
from use.timeseries import TimeSeries
import use.MatrixProfile as mp
import use.similarity_measures as sm
from variableQueryMP.iterationData import IterationData

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

def computeMP(timeseries1, timeseries2, m, skip_step):
    #timeseries1: Query TS, timeseries2: Target TS
    iterationDataList = []

    t1 = timeseries1
    t2 = timeseries2
    n1 = len(t1.timeseries)
    n2 = len(t2.timeseries)
    indexes = n1 - m + 1
    MP12 = [] #Matrix Profile
    #IP12 = [0] #Index Profile
    DP_all = {} # Distance Profiles for All Index in the timeseries
    idx = 0
    if int(m/skip_step)==0:
        step = 1
    else:
        step = int(m / skip_step)
    for index in range(0, indexes, step):
        data = t2.timeseries
        index2 = index + m
        #query = t2.timeseries[index:index2]
        query = t1.timeseries[index:index2]
        # compute Distance Profile(DP)
        #DP = mass_v2(data, query)
        # if std(query)==0, then 'mass_v2' will return a NAN, ignore this Distance profile
        #Numpy will generate the result with datatype 'float64', where std(query) maybe equals to 'x*e-17', but not 0
        if round(np.std(query),4) == 0:
            continue
        else:
            DP_all[idx], meanQ, sigmaQ, meanT, sigmaT, QT = sm.mass_v2(data, query)
            MP12.append(min(DP_all[idx]))
            idx += 1
            iterationData = IterationData()
            #Q is a subsequence, T is the entire timeseries, meanT/sigmaT are lists of elements
            meanQplus, sigmaQplus = iterationData.updateMeanSigma(meanQ, sigmaQ, query)
            iterationData.LB = sm.computeLB(QT, m, meanQ, meanT, sigmaQ, sigmaT, sigmaQplus)
            iterationData.meanQ = meanQ
            iterationData.sigmaQ = sigmaQ
            iterationData.meanT = meanT
            iterationData.sigmaT = sigmaT
            iterationData.QT = QT
            iterationDataList.append(iterationData)
    return DP_all, MP12, iterationDataList

# newL here is L+1, the incremental step is 1 by default
def updateMP(TS_Query, TS_Target, IterationDataList, new_m, skip_step):
    n_Q = len(TS_Query)
    if int(new_m/skip_step)==0:
        step = 1
    else:
        step = int(new_m / skip_step)
    nbr_offset = int((n_Q - new_m) / step)
    DP = {}
    MP = {}
    IterationDataNewList = {}
    for i in range(0, nbr_offset):
        index = i * step
        Q = TS_Query[index:index + new_m]
        iterationData = IterationDataList[i]
        '''lb_dict = LB[i]
        lb_list = [(idx, dist) for idx, dist in lb_dict.items()]
        lb_list = sorted(lb_list, key=lambda d: d[1])
        dp, min_dist, meanQ[i], sigmaQ[i], meanT[i], sigmaT[i], QT[i], flag = updateMpPart(lb_list, QT[i], Q, TS_Target,
                                                                                       meanQ[i], sigmaQ[i],
                                                                                       meanTList=meanT[i],
                                                                                       sigmaTList=sigmaT[i])'''
        dp, min_dist, iterationData = updateMpPart(Q, TS_Target, iterationData)
        DP[i] = dp
        MP[i] = min_dist
        IterationDataNewList[i] = iterationData
    return DP, MP, IterationDataNewList

# from the given data of length =l, to get the minimal distance and Distance Profile for length =l+1
# Q(length = l+1), T; QT, meanQ, sigmaQ, meanTList, sigmaTList:(length = l)
#def updateMpPart(lb_list, QT, Q, T, meanQ, sigmaQ, meanTList, sigmaTList):
def updateMpPart(Q, T, iterationData):
    meanT = iterationData.meanT
    sigmaT = iterationData.sigmaT
    meanQ = iterationData.meanQ
    sigmaQ = iterationData.sigmaQ
    QT = iterationData.QT
    lb_list = iterationData.LB

    meanQplus, sigmaQplus = iterationData.updateMeanSigma(meanQ, sigmaQ, Q)
    meanTplus = {}
    sigmaTplus = {}
    QTplus = {}
    #for query of length=l-1, the number of mappings decreases by 1
    nbr_distance = len(lb_list) -1
    L = len(Q)
    DPplus = {}
    dist_list = []
    min_dist = None
    for i, (rawIdx, lb) in enumerate(lb_list):
        # check if rawIdx exists in the original list
        T_subseq = T[rawIdx : rawIdx+L]
        meanTplus[rawIdx], sigmaTplus[rawIdx], QTplus[rawIdx] = iterationData.updateParaT(rawIdx, meanT[rawIdx], sigmaT[rawIdx], Q, T_subseq, QT[rawIdx])
        dist_exact = sm.compute1Dist(meanQplus, meanTplus[rawIdx], sigmaQplus, sigmaTplus[rawIdx], QTplus[rawIdx], L)
        index, rawIndexList = locateDistIndex(dist_exact, lb_list)
        if index is None or index+i >= nbr_distance:
            DPplus.update({rawIdx: dist_exact})
            continue
        else:
            # have found an exact distance in LB profile
            for idx in rawIndexList:
                # Two choses: - try to use the RAW index, or use the index in 'lb_list'
                #retrun to the original index in 'lb_list': QT, mean, sigma
                meanTplus[idx], sigmaTplus[idx], QTplus[idx] = iterationData.updateParaT(idx, meanT[idx], sigmaT[idx], Q, T_subseq, QT[idx])
                dist_exact = sm.compute1Dist(meanQplus, meanTplus[idx], sigmaQplus, sigmaTplus[idx], QTplus[idx], L)
                DPplus.update({idx: dist_exact})
                dist_list.append(dist_exact)
            min_dist = min(dist_list)
            break
    if min_dist is None:
        min_dist = min(DPplus.values())
        # limite the number of element in LBplus, decrease 1
        LBplus = sm.computeLB(QT, L, meanQ, meanT, sigmaQ, sigmaT, sigmaQplus)
        lb_list = [(idx, dist) for idx, dist in LBplus.items()]
        lb_list = sorted(lb_list, key=lambda d: d[1])
        iterationData.LB = lb_list
    else:
        LBplus = sm.linearComputeLB(lb_list, sigmaQ, sigmaQplus)
        iterationData.LB = LBplus
    iterationData.meanQ = meanQplus
    iterationData.sigmaQ = sigmaQplus
    iterationData.meanT = meanTplus
    iterationData.sigmaT = sigmaTplus
    iterationData.QT = QTplus

    return DPplus, min_dist, iterationData

# to check the existence of dist in LB profile
def locateDistIndex(dist_exact, lb_list):
    #lb_list: [(rawIndex, LB)]
    for idx in range(0, len(lb_list)):
        low_value = lb_list.__getitem__(idx - 1)[1]
        high_value = lb_list.__getitem__(idx + 1)[1]
        if low_value <= dist_exact and high_value >= dist_exact:
            rawIndexList = []
            for i in range(0, idx):
                rawIndexList.append(lb_list.__getitem__(idx)[0])
            return idx, rawIndexList
    return None, None
