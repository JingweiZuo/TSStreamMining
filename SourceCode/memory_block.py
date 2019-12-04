import numpy as np
import SMAP.MatrixProfile as mp

# centralized & distributed algorithm
def TS_cache(TS_set_input, stack_size, inputTSBatch):
    #cache TS batches
    TS_set = TS_set_input.copy()
    TS_set.extend(inputTSBatch)
    if len(TS_set) > stack_size:
        TS_set_new = TS_set[-stack_size:].copy()
        return TS_set_new
    return TS_set

def MP_cache(TS_set_input, MP_set_input, stack_size, inputTSBatch, m, distance_measure):
    #cache Matrix Profile
    MP_set = MP_set_input.copy()
    i = 0
    j = 0
    w = len(inputTSBatch)
    TS_set = TS_set_input.copy()
    if len(TS_set) == 0:
        MP_set = np.empty((w, w),dtype=object)
        for ts_source in inputTSBatch:
            for ts_target in inputTSBatch:
                if ts_source!=ts_target :
                    MP_set[i][j] = mp.computeMP(ts_source, ts_target, m, distance_measure)
                j += 1
            i += 1
            j = 0
    else:
        row = np.empty((w, len(MP_set)),dtype=object)
        col = np.empty((w, len(MP_set)+w),dtype=object)
        MP_set = np.insert(MP_set, len(MP_set), row, axis=0)
        MP_set = np.insert(MP_set, len(MP_set[1]), col, axis=1)
        TS_set_old = TS_set.copy()
        TS_set.extend(inputTSBatch)
        for ts_old in TS_set:
            for ts_new in inputTSBatch:
                if ts_old != ts_new :
                    MP_set[i][len(TS_set_old)+j] = mp.computeMP(ts_old, ts_new, m, distance_measure)
                    MP_set[len(TS_set_old)+j][i] = mp.computeMP(ts_new, ts_old, m, distance_measure)
                j += 1
            i += 1
            j = 0
        # fill the table between ts_new and ts_new
        i = 0
        j = 0
        for ts_new_source in inputTSBatch:
            for ts_new_target in inputTSBatch:
                if ts_old != ts_new:
                    MP_set[len(TS_set_old) + i][len(TS_set_old) + j] = mp.computeMP(ts_new_source, ts_new_target, m, distance_measure)
                    MP_set[len(TS_set_old) + j][len(TS_set_old) + i] = mp.computeMP(ts_new_target, ts_new_source, m, distance_measure)
                j += 1
            i += 1
            j = 0
    if len(MP_set[0]) > stack_size: # Test is OK
        MP_set = MP_set[-stack_size:,-stack_size:]
    return MP_set

def memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure):
    for m in m_list:
        if m not in MP_set_all.keys():  # Test is OK
            w = len(inputTSBatch)
            MP_set = np.empty((w, w), dtype=object)
            MP_set_all.update({m: MP_cache(TS_set, MP_set, stack_size, inputTSBatch, m, distance_measure)})
        else:
            MP_set_all.update({m: MP_cache(TS_set, MP_set_all[m], stack_size, inputTSBatch, m, distance_measure)})
    TS_newSet = TS_cache(TS_set, stack_size, inputTSBatch)
    return TS_newSet, MP_set_all

#kick-off event: transition from Concept Drift to Stable Concept
def elastic_caching_mechanism(cached_TS, cached_MP, shapList, window_size, driftDetection, drift_strategy):
    # Stop point of caching elimination process: when there's no Concept Drift
    # resolve the index for eliminated historical data
    cached_TS, cached_MP = driftDetection.eliminate_caching(cached_TS, cached_MP, shapList, window_size, drift_strategy)
    return cached_TS, cached_MP
