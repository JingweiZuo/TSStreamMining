import numpy as np

# for the data of an offset
class IterationData(object):
    def __init__(self):
        self.QT = {}
        self.LB = {}
        self.meanQ = None
        self.sigmaQ = None
        self.meanT = {}
        self.sigmaT = {}

    def updateMeanSigma(mean, sigma, TS):
        return 0

    def updateQT(self, QT, Q, T_subseq):
        return 0

    #def updateParaT(self, rawIdx, meanTList, sigmaTList, T_subseq, QT, Q):
    def updateParaT(self, rawIdx, meanT, sigmaT, Q, T_subseq, QT):
        meanTplus = None
        sigmaTplus = None
        QTplus = None
        if rawIdx in meanT.keys():
            meanTplus, sigmaTplus = self.updateMeanSigma(meanT[rawIdx], sigmaT[rawIdx], T_subseq)
            QTplus = self.updateQT(QT[rawIdx], Q, T_subseq)
        else:
            meanTplus = np.mean(T_subseq)
            sigmaTplus = np.std(T_subseq)
            QTplus = np.dot(Q, T_subseq)
        '''if rawIdx in QT.keys():
            QTplus = updateQT(QT[rawIdx], Q, T_subseq)
        else:
            QTplus = np.dot(Q, T_subseq)'''
        return meanTplus, sigmaTplus, QTplus