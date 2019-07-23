import numpy as np
import line_profiler


def sliding_window(sequence, win_size, step=1):
    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(win_size) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > win_size:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if win_size > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    num_chunks = ((len(sequence) - win_size) / step) + 1

    # Do the work
    for i in range(0, int(num_chunks * step), step):
        yield sequence[i:i + win_size]


def calculate_distances(timeseries, subsequence, distance_measure):
    if distance_measure == "mass_v1":
        return mass_v1(subsequence, timeseries)
    elif distance_measure == "mass_v2":
        return mass_v2(timeseries, subsequence)
    elif distance_measure == "brute":
        return euclidean_distance_unequal_lengths(timeseries, subsequence)


def euclidean_distance(t1, t2):
    return np.sqrt(sum((t1 - t2) ** 2))


def euclidean_distance_unequal_lengths(t, s):  ##O(m)
    ## return a array of distance between 'shapelet' and every slices of 'timeseries'
    distances = np.array([euclidean_distance(np.array(s1), s) for s1 in sliding_window(t, len(s))])
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

# v1: Full matching MASS, basic version
def mass_v1(q, t):
    m, n = len(q), len(t)
    # Z-normalization of Query
    q = (q - np.mean(q)) / np.std(q)
    qt = dot_products_1(q, t)
    # compute the mean and standard deviation of Time Series
    sum_q = np.sum(q)
    sum_q2 = np.sum(np.power(q, 2))

    # cache a cumulative sum of values
    cum_sum_t = np.cumsum(t)
    cum_sum_t2 = np.cumsum(np.power(t, 2))

    # sum of x and x square for [0, n-m] subsequences of length m
    # sumt2 = cum_sum_t2[m-1:] - cum_sum_t2[:- m+1]
    # sumt = cum_sum_t[m-1:] - cum_sum_t[:- m+1]
    sumt2 = cum_sum_t2[m:] - cum_sum_t2[: n - m]
    sumt = cum_sum_t[m:] - cum_sum_t[: n - m]
    meant = sumt / m
    # standard deviation of every subsequence of length m
    sigmat2 = (sumt2 / m) - (np.power(meant, 2))
    sigmat = np.sqrt(sigmat2)

    dist = (sumt2 - 2 * sumt * meant + m * (np.power(meant, 2))) / sigmat2 - 2 * (
                qt[m:n] - sum_q * meant) / sigmat + sum_q2
    dist = np.sqrt(dist)
    # distance here is a complex number, need to return its amplitude/absolute value
    # return a vector with size of n-m+1
    return np.abs(dist)


# v2: Half matching MASS, advanced version
def mass_v2(x, y):
    # x is the data, y is the query
    n, m = len(x), len(y)

    # %compute y stats -- O(n)
    meany = np.mean(y)
    sigmay = np.std(y)

    # compute x stats -- O(n)
    # compute the moving average and standard deviation of Time Series
    meanx = running_mean(x, m)
    sigmax = running_std(x, m)

    # The main trick of getting dot products in O(n log n) time
    z = dot_products_2(y, x)
    dist = 2 * (m - (z[m - 1:n] - m * meanx[m - 1:n] * meany) / (sigmax[m - 1:n] * sigmay))
    dist = np.sqrt(dist)
    # distance here is a complex number, need to return its amplitude/absolute value
    # return a vector with size of n-m+1
    return np.abs(dist)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, np.zeros(N)))
    cum_v = cumsum[N:] - cumsum[:-N]
    cum_v1 = np.divide(cum_v[:N],range(1,N+1))
    cum_v2 = cum_v[N:]/ float(N)
    return np.concatenate([cum_v1,cum_v2])

def running_std(x, N):
    x2 = np.power(x, 2)
    cumsum2 = np.cumsum(np.insert(x2, 0, np.zeros(N)))
    cum_v = cumsum2[N:] - cumsum2[:-N]
    cum_v1 = np.divide(cum_v[:N],range(1,N+1))
    cum_v2 = cum_v[N:]/ float(N)
    cumstd2 = np.concatenate([cum_v1,cum_v2])
    return (cumstd2 - running_mean(x, N) ** 2) ** 0.5

# v3: advanced version using historical computations
def mass_v3(x, y, meanx, meany, sigmax, sigmay, sigmaQplus):
    # x is the data, y is the query
    n, m = len(x), len(y)

    # The main trick of getting dot products in O(n log n) time
    z = dot_products_2(y, x)
    # dist = 2 * m * (1 - (z[m-1:n]/m - meanx[m-1:n] * meany) / (sigmax[m-1:n] * sigmay))
    # dist = np.sqrt(dist)
    # distance here is a complex number, need to return its amplitude/absolute value
    # return a vector with size of n-m+1

    sigmaxy = sigmax * sigmay
    q_ij = (z[m - 1:n] / m - meanx * meany) / sigmaxy
    dist = np.sqrt(2 * m * (1 - q_ij))

    # Q is a subsequence, T is the entire timeseries, meanT/sigmaT are lists of elements
    coeff = sigmay / sigmaQplus
    # compute the LB profile: q_ij
    # q_ij[q_ij <= 0] = (m ** 0.5) * coeff
    # q_ij[q_ij > 0] = ((m * (1 - q_ij ** 2)) ** 0.5) * coeff
    LB = np.abs(
        np.where(q_ij <= 0, (m ** 0.5) * coeff, np.where(q_ij > 0, ((m * (1 - q_ij ** 2)) ** 0.5) * coeff, q_ij)))
    qt = {idx: value for idx, value in enumerate(z[m - 1:n])}

    lb_list = [(idx, dist) for idx, dist in enumerate(LB)]
    lb_list = sorted(lb_list, key=lambda d: d[1])
    return np.abs(dist), qt, lb_list

# @profile
def compute1Dist(meanQ, meanT, sigmaQ, sigmaT, QT, m):
    if sigmaT <= 0.0001:
        dist = 10000
    else:
        dist = 2 * (m - (QT - m * meanQ * meanT) / (sigmaQ * sigmaT))
    return abs(dist * 0.5)

def linearComputeLB(LB, sigmaQ, sigmaQplus):
    # LB: [(rawIndex, value)]
    LB_new = [(rawIndex, value * sigmaQ / sigmaQplus) for (rawIndex, value) in LB]
    return LB_new

# @profile
def computeLB(QT, m, meanQ, meant, sigmaQ, sigmat, sigmaQplus):
    # Q is a subsequence, T is an entire timeseries
    qt = np.array([QT[i] for i in sorted(QT)])
    lenQT = len(qt)
    # print("length of QT is ", str(lenQT), "length of meant is ", str(len(meant)))
    # problem of LB, as when sigma<0.0001, skip
    # q_ij = (qt/m - meanQ*meant[:lenQT]) / sigmaQ * sigmat[:lenQT]
    q_ij = (qt / m - meanQ * meant) / sigmaQ * sigmat
    coeff = sigmaQ / sigmaQplus
    q_ij = np.abs(np.where(q_ij.real <= 0, (m ** 0.5) * coeff,
                           np.where(q_ij.real > 0, ((m * (1 - q_ij.real ** 2)) ** 0.5) * coeff, q_ij)))
    lb_list = [(idx, dist) for idx, dist in enumerate(q_ij)]
    lb_list = sorted(lb_list, key=lambda d: d[1])
    return lb_list


# @profile
def computeMeanSigma(dataset, m):
    mean = {}
    sigma = {}
    for ts in dataset.values():
        TS = ts.timeseries
        n = len(TS)
        # compute the moving average and standard deviation of Time Series
        mean_ts = running_mean(TS, n)
        sigma_ts = running_std(TS, n)
        mean[ts.name] = mean_ts[m - 1:n]
        sigma[ts.name] = sigma_ts[m - 1:n]
    return mean, sigma


# @profile
# input: mean(m), sigma(m), mplus(m+1)
# output: mean_new(m+1), sigma_new(m+1)
def updateMeanSigma(dataset, mean, sigma, mplus):
    mean_new = {}
    sigma_new = {}
    for ts in dataset.values():
        TS = ts.timeseries
        n = len(TS)
        new_elem = TS[mplus - 1:]
        mean_temp = mean[ts.name][:n - mplus + 1]
        mean_new[ts.name] = (mean_temp * (mplus - 1) + new_elem) / mplus

        sigma_temp = sigma[ts.name][:n - mplus + 1]
        ss_old = (mplus - 1) * (sigma_temp ** 2 + mean_temp ** 2)
        ss_new = ss_old + new_elem ** 2
        sigma_new[ts.name] = ((ss_new / mplus) - mean_new[ts.name] ** 2) ** 0.5
    return mean_new, sigma_new