import numpy as np

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

def mass_v2(x, y):
    #x is the data, y is the query
    n, m = len(x), len(y)

    #%compute y stats -- O(n)
    meany = np.mean(y);
    sigmay = np.std(y);

    #compute x stats -- O(n)
    #compute the average of the first m elements in 'x'
    def running_mean(x, N):
        cumsum = numpy.cumsum(numpy.insert(x, 0, np.zeros(N)))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def running_std(x, N):
        x2 = np.power(x, 2)
        cumsum2 = numpy.cumsum(numpy.insert(x2, 0, np.zeros(N)))
        return (cumsum2[N:] - cumsum2[:-N]) / float(N) - running_mean(x, N)

    #compute the moving average and standard deviation of Time Series 
    meanx = running_mean(x, n)
    sigmax = running_std(x, n)

    #The main trick of getting dot products in O(n log n) time
    z = dot_products_2(y, x)
    dist = 2*(m-(z[m-1:n]-m*meanx[m-1:n]*meany)/(sigmax[m-1:n]*sigmay))
    dist = np.sqrt(dist);
    #distance here is a complex number, need to return its amplitude/absolute value
    #return a vector with size of n-m+1
    return np.abs(dist)
