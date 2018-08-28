import numpy as np
from utils.old_Utils import old_Utils
import line_profiler
from variableQueryMP.iterationData import IterationData

def calculate_distances(timeseries, subsequence, distance_measure):
    if distance_measure == "mass_v1" :
        return mass_v1(subsequence, timeseries)
    elif distance_measure == "mass_v2" :
        return mass_v2(timeseries, subsequence)
    elif distance_measure == "brute":
        return euclidean_distance_unequal_lengths(timeseries, subsequence)

def euclidean_distance(t1, t2):
    return np.sqrt(sum((t1 - t2) ** 2))

def euclidean_distance_unequal_lengths(t, s):##O(m)
    ## return a array of distance between 'shapelet' and every slices of 'timeseries'
    distances = np.array([euclidean_distance(np.array(s1), s) for s1 in old_Utils.sliding_window(t, len(s))])
    return distances

def dot_products_1(q, t):
    m, n = len(q), len(t)
    t_a = np.concatenate([t, np.zeros(n)])
    # reverse the Query
    q_r = q[::-1]
    q_ra = np.concatenate([q_r, np.zeros(2 * n - m)])
    q_raf = np.fft.fft(q_ra)
    t_af = np.fft.fft(t_a)
    qt = np.fft.ifft(q_raf * t_af)
    return qt

def dot_products_2(q, t):
    # concatenante n-m zeros for Query
    m, n = len(q), len(t)
    # reverse the Query
    q_r = q[::-1]
    q_ra = np.concatenate([q_r, np.zeros(n - m)])
    q_raf = np.fft.fft(q_ra)
    t_af = np.fft.fft(t)
    qt = np.fft.ifft(q_raf * t_af)
    return qt

def mass_v1(q, t):
    m, n = len(q), len(t)
    # Z-normalization of Query
    q = (q-np.mean(q)) / np.std(q)
    qt = dot_products_1(q, t)
    #compute the mean and standard deviation of Time Series
    sum_q = np.sum(q)
    sum_q2 = np.sum(np.power(q,2))

    #cache a cumulative sum of values
    cum_sum_t = np.cumsum(t)
    cum_sum_t2 = np.cumsum(np.power(t,2))

    #sum of x and x square for [0, n-m] subsequences of length m
    sumt2 = cum_sum_t2[m-1:] - cum_sum_t2[:- m+1]
    sumt = cum_sum_t[m-1:] - cum_sum_t[:- m+1]
    meant = sumt / m
    #standard deviation of every subsequence of length m
    sigmat2 = (sumt2 / m) - (np.power(meant,2))
    sigmat = np.sqrt(sigmat2)

    dist = (sumt2 - 2 * sumt * meant + m * (np.power(meant,2))) / sigmat2 - 2 * (qt[m-1:n] - sum_q * meant) / sigmat + sum_q2
    dist = np.sqrt(dist)
    #distance here is a complex number, need to return its amplitude/absolute value
    #return a vector with size of n-m+1
    return np.abs(dist)

@profile
def mass_v2(x, y):
    #x is the data, y is the query
    n, m = len(x), len(y)

    #%compute y stats -- O(n)
    meany = np.mean(y)

    sigmay = np.std(y)
    #compute x stats -- O(n)
    #compute the average of the first m elements in 'x'
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, np.zeros(N)))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def running_std(x, N):
        x2 = np.power(x, 2)
        cumsum2 = np.cumsum(np.insert(x2, 0, np.zeros(N)))
        return ((cumsum2[N:] - cumsum2[:-N]) / float(N) - running_mean(x, N) ** 2) ** 0.5

    #compute the moving average and standard deviation of Time Series
    meanx = running_mean(x, n)
    sigmax = running_std(x, n)

    #The main trick of getting dot products in O(n log n) time
    z = dot_products_2(y, x)
    dist = 2 * (m - (z[m-1:n] - m * meanx[m-1:n] * meany) / (sigmax[m-1:n] * sigmay))
    dist = np.sqrt(dist)
    #distance here is a complex number, need to return its amplitude/absolute value
    #return a vector with size of n-m+1
    return np.abs(dist)

@profile
def mass_v3(x, y, Qplus):
    #x is the data, y is the query
    n, m = len(x), len(y)

    #%compute y stats -- O(n)
    meany = np.mean(y)

    sigmay = np.std(y)
    #compute x stats -- O(n)
    #compute the average of the first m elements in 'x'
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, np.zeros(N)))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def running_std(x, N):
        x2 = np.power(x, 2)
        cumsum2 = np.cumsum(np.insert(x2, 0, np.zeros(N)))
        return ((cumsum2[N:] - cumsum2[:-N]) / float(N) - running_mean(x, N) ** 2) ** 0.5

    #compute the moving average and standard deviation of Time Series
    meanx = running_mean(x, n)
    sigmax = running_std(x, n)

    #The main trick of getting dot products in O(n log n) time
    z = dot_products_2(y, x)
    #dist = 2 * m * (1 - (z[m-1:n]/m - meanx[m-1:n] * meany) / (sigmax[m-1:n] * sigmay))
    #dist = np.sqrt(dist)
    #distance here is a complex number, need to return its amplitude/absolute value
    #return a vector with size of n-m+1

    sigmaxy = sigmax[m - 1:n] * sigmay
    q_ij = (z[m-1:n] / m - meanx[m-1:n] * meany) / sigmaxy
    dist = np.sqrt(2 * m * (1 - q_ij))

    iterationData = IterationData()
    # Q is a subsequence, T is the entire timeseries, meanT/sigmaT are lists of elements
    meanQplus, sigmaQplus = iterationData.updateMeanSigma(meany, sigmay, Qplus)
    coeff = sigmay / sigmaQplus
    # compute the LB profile: q_ij
    # q_ij[q_ij <= 0] = (m ** 0.5) * coeff
    # q_ij[q_ij > 0] = ((m * (1 - q_ij ** 2)) ** 0.5) * coeff
    LB = np.where(q_ij <= 0, (m ** 0.5) * coeff, np.where(q_ij > 0, ((m * (1 - q_ij ** 2)) ** 0.5) * coeff, q_ij))


    meant = {idx: value for idx, value in enumerate(meanx)}
    sigmat = {idx: value for idx, value in enumerate(sigmax)}
    qt = {idx: value for idx, value in enumerate(z)}

    lb_list = [(idx, dist) for idx, dist in enumerate(LB)]
    lb_list = sorted(lb_list, key=lambda d: d[1])
    iterationData.LB = lb_list
    iterationData.meanQ = meany
    iterationData.sigmaQ = sigmay
    iterationData.meanT = meant
    iterationData.sigmaT = sigmat
    iterationData.QT = qt
    return np.abs(dist), iterationData #q_ij here is the LB profile

'''to complete'''
@profile
def compute1Dist(meanQ, meanT, sigmaQ, sigmaT, QT, m):
    if sigmaT <= 0.0001:
        dist = 10000
    else:
        dist = 2 * (m - (QT - m * meanQ * meanT) / (sigmaQ * sigmaT))
    return abs(dist*0.5)

def linearComputeLB(LB, sigmaQ, sigmaQplus):
    #LB: [(rawIndex, value)]
    LB_new = [(rawIndex, value* sigmaQ / sigmaQplus) for (rawIndex,value) in LB]
    return LB_new

@profile
def computeLB(QT, m, meanQ, meanT, sigmaQ, sigmaT, sigmaQplus):
    # Q is a subsequence, T is an entire timeseries
    qt = np.array(list(QT.values()))
    meant = np.array(list(meanT.values()))
    sigmat = np.array(list(sigmaT.values()))
    q_ij = (qt/m - meanQ*meant) / sigmaQ * sigmat
    LB = {}
    coeff= sigmaQ / sigmaQplus
    q_ij = np.where(q_ij <= 0, (m ** 0.5) * coeff, np.where(q_ij > 0, ((m * (1 - q_ij ** 2)) ** 0.5) * coeff, q_ij))
    lb_list = [(idx, dist) for idx, dist in enumerate(q_ij)]
    lb_list = sorted(lb_list, key=lambda d: d[1])

    '''for idx, val in enumerate(q_ij):
        if val <= 0:
            LB[idx] = (L**0.5) * coeff
        else:
            LB[idx] = ((L * (1 - val**2))**0.5) * coeff'''
    return lb_list
