import numpy as np
from use.timeseries import TimeSeries
import use.MatrixProfile as mp
import use.similarity_measures as sm

def computeIterateMP(TS_Query, TS_Target, L, newL, step):
    DP = {}
    MP = {}
    QT = {}
    LB = {}
    meanQ = {}
    sigmaQ = {}
    meanT = {}
    sigmaT = {}

    #Initalization
    n_Q = len(TS_Query)
    n_T = len(TS_Target)
    #meanT[L] here is a list of list, same as DP[L]
    DP[L], MP[L], QT[L], meanQ[L], sigmaQ[L], meanT[L], sigmaT[L] = mp.computeMP(TS_Query, TS_Target, L, step)
    LB[L+1] = sm.computeLbGlobal(DP[L])

    for l in range(L, newL):
        nbr_offset = int((n_Q - l) / step)
        for i in range(0, nbr_offset):
            index = nbr_offset * step
            Q = TS_Query[index:index+l]

            # compute the Sigma and the mean of Query
            #LB:{length: lb_unit}, lb_unit:{offset:lb_temp}, lb_dict: {rawIndex: LbDistance}
            lb_dict = LB[l+1][i]
            lb_list = [(idx, dist) for idx, dist in lb_dict.items()]
            lb_list = sorted(lb_list, key = lambda d:d[1])
            min_dist, dp, qt, flag, mean, sigma = updateMP(lb_list, QT[l][i], Q, TS_Target, meanQ[L][i], sigmaQ[L][i])
            '''Stop here'''
            MP[l+1][i] = min_dist
            DP[l+1][i] = dp
            QT[l+1][i] = qt
            meanQ[L+1][i] = mean
            sigmaQ[L+1][i] = sigma
            meanQ.pop(L)
            sigmaQ.pop(L)

            if flag:
                LB[l+2][i] = sm.linearComputeLB(LB[l+1][i], sigmaQ)
            else:
                LB[l+2][i] = sm.computeLB(QT, L, meanQ[L][i], meanT[L][i], sigmaQ[L][i], sigmaT[L][i], sigmaQ[L+1][i])
    return MP

# from the given data of length =l, to get the minimal distance and Distance Profile for length =l+1
# Q(length = l+1), T; QT, meanQ, sigmaQ, meanTList, sigmaTList:(length = l)
def updateMP(lb_list, QT, Q, T, meanQ, sigmaQ, meanTList, sigmaTList):
    meanQplus, sigmaQplus = updateMeanSigma(meanQ, sigmaQ, Q)
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
        meanTplus[rawIdx], sigmaTplus[rawIdx], QTplus[rawIdx] = updateParaT(rawIdx, meanTList, sigmaTList, T_subseq, QT, Q)
        dist_exact = sm.compute1Dist(meanQplus, meanTplus[rawIdx], sigmaQplus, sigmaTplus[rawIdx], QTplus[rawIdx], L)
        index = locateDistIndex(dist_exact, lb_list)
        if index is None or index+i >= nbr_distance:
            DPplus.update(rawIdx, dist_exact)
            continue
        else:
            # have found an exact distance in LB profile
            rawIndexList = sm.getRawIndexLB(lb_list, index)
            for idx in rawIndexList:
                # Two choses: - try to use the RAW index, or use the index in 'lb_list'
                #retrun to the original index in 'lb_list': QT, mean, sigma
                meanTplus[idx], sigmaTplus[idx], QTplus[idx] = updateParaT(idx, meanTList, sigmaTList, T_subseq, QT, Q)
                dist_exact = sm.compute1Dist(meanQplus, meanTplus[idx], sigmaQplus, sigmaTplus[idx], QTplus[idx], L)
                DPplus.update(idx, dist_exact)
                dist_list.append(dist_exact)
            min_dist = min(dist_list)
            break
    if min_dist is None:
        min_dist = min(DPplus.values())
        return DPplus, min_dist, meanQplus, sigmaQplus, meanTplus, sigmaTplus, False
    else:
        return DPplus, min_dist, meanQplus, sigmaQplus, meanTplus, sigmaTplus, True
# to check the existence of dist in LB profile

def updateMeanSigma(mean, sigma, TS):
    return 0

def updateQT(QT_old, Q, T_subseq):
    return 0

def updateParaT(rawIdx, meanTList, sigmaTList, T_subseq, QT, Q):
    meanTplus = None
    sigmaTplus = None
    QTplus = None
    if rawIdx in meanTList.keys():
        meanTplus, sigmaTplus = updateMeanSigma(meanTList[rawIdx], sigmaTList[rawIdx], T_subseq)
        QTplus = updateQT(QT[rawIdx], Q, T_subseq)
    else:
        meanTplus = np.mean(T_subseq)
        sigmaTplus = np.std(T_subseq)
        QTplus = np.dot(Q, T_subseq)
    '''if rawIdx in QT.keys():
        QTplus = updateQT(QT[rawIdx], Q, T_subseq)
    else:
        QTplus = np.dot(Q, T_subseq)'''
    return meanTplus, sigmaTplus, QTplus

def locateDistIndex():
    return 0
