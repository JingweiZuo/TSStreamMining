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

    def updateMeanSigma(self, mean, sigma, T_subseq):
        m_new = len(T_subseq)
        m = m_new - 1
        mean_new = (mean * m + T_subseq[-1]) / m_new
        ss_old = m * (sigma**2 + mean**2)
        ss_new = ss_old + T_subseq[-1]**2
        sigma_new = ((ss_new/m_new) - mean_new**2)**0.5
        return mean_new, sigma_new

    def updateQT(self, QT, Q, T_subseq):
        QT_new = QT + Q[-1]*T_subseq[-1]
        return QT_new

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