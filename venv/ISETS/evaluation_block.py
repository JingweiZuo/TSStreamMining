import numpy as np
import utils.similarity_measures as sm
import utils.utils as util

class driftDetection(object):
    def __init__(self):
        self.n_batch = 0
        self.t_stamp = 0
        self.avg_loss = 0
        self.cum_loss = 0
        self.mincum_loss = np.inf
        self.theta = 1  # The slope of Sigmoid Function

        # Parameters for PH test of Concept Drift
        self.tolerance = 0  # The loss tolerance for PH test
        self.thresh = 0  # Threshold of concept drift for PH test

        # Parameters for simple detection of Concept Drift
        self.thresh_loss = 0.5

    #sigmoid = lambda x: 1 / (1 + np.exp(-x))
    def sigmoid(self, x, theta, x0):
        return (1 / (1 + np.exp(-theta*(x-x0))))

    def ConceptDrift_detection(self, shap_set, TS_window, drift_strategy):
        # a manually setting of loss threshold (0, 1), where 0.5 complies to 0-1 loss function
        drift = True
        loss_batch = 0
        w = len(TS_window)
        for ts in TS_window:
            shap_set_part = [s for s in shap_set if s.Class == ts.class_timeseries]
            #if len(shap_set) == 0:
                #print("MARKER_EB: shap_set is empty")
            min_dist = np.inf
            min_s = util.Shapelet()
            # find the closest Shapelet to 'ts'
            for s in shap_set_part:
                if round(np.std(s.subseq),4) != 0:
                    dist = np.min(sm.calculate_distances(ts.timeseries, s.subseq, "mass_v2"))
                    if min_dist > dist:
                        min_dist = dist
                        min_s = s
            #print("MARKER_EB: min_dist between " + str(ts.id) + "and NN Shapelet is: " + str(min_dist))
            l = self.sigmoid(min_dist, self.theta, min_s.dist_threshold)
            loss_batch += l
        loss_batch = loss_batch / w
        self.n_batch += 1
        if drift_strategy == "manual_set loss":
            # Method 1: set manually a loss threshold, once the detected batch loss exceeds the threshold, then a drift is detected
            if loss_batch <= self.thresh_loss:
                drift = False
            return drift, loss_batch
        elif drift_strategy == "mean loss variance":
            # Method 2: when batch loss exceeds the global average loss, a drift is detected
            # How to determine the global average loss? -> use the same average loss computed until current time tick
            self.avg_loss = self.avg_loss * (self.n_batch - 1) / self.n_batch + loss_batch / self.n_batch
            if loss_batch <= self.avg_loss:
                drift = False
            return drift, loss_batch, self.avg_loss
        else:
            # Method 3: when PH test value exceeds a manual-set threshold, a drift is detected
            self.avg_loss = self.avg_loss * (self.n_batch - 1) / self.n_batch + loss_batch / self.n_batch
            self.cum_loss = self.cum_loss + loss_batch - self.avg_loss - self.tolerance
            PH = self.cum_loss - self.mincum_loss
            '''if self.n_batch == 1:
                        self.cum_loss = self.avg_loss'''
            if PH < 0:
                self.mincum_loss = self.cum_loss
            if PH <= self.thresh:
                drift = False
            return drift, loss_batch, self.avg_loss, self.cum_loss, self.mincum_loss, PH

    def eliminate_caching(self, cached_TS, cached_MP, shap_set, window_size, drift_strategy):
        # cached_data: DP/MP, TS instances
        drift = True
        elim_num = 0
        # until a transition point is found in cached data
        while drift:
            TS_window = cached_TS[elim_num * window_size: elim_num * window_size + window_size]
            loss_batch = 0
            w = len(TS_window)
            for ts in TS_window:
                shap_set_part = [s for s in shap_set if s.Class == ts.class_timeseries]
                min_dist = np.inf
                min_s = util.Shapelet()
                # find the closest Shapelet to 'ts'
                for s in shap_set_part:
                    dist = np.min(sm.calculate_distances(ts.timeseries, s.subseq, "mass_v2"))
                    if min_dist > dist:
                        min_dist = dist
                        min_s = s
                l = self.sigmoid(min_dist, self.theta, min_s.dist_threshold)
                loss_batch += l
            loss_batch = loss_batch / w
            if drift_strategy == "manual_set loss":
                # Method 1: set manually a loss threshold, once the detected batch loss exceeds the threshold, then a drift is detected
                if loss_batch <= self.thresh_loss:
                    drift = False
            elif drift_strategy == "mean loss variance":
                # Method 2: when batch loss exceeds the global average loss, a drift is detected
                # How to determine the global average loss? -> use the same average loss computed until current time tick
                avg_loss_temp = self.avg_loss * (self.n_batch - 1) / self.n_batch + loss_batch / self.n_batch
                if loss_batch <= avg_loss_temp:
                    drift = False
            else:
                # Method 3: when PH test value exceeds a manual-set threshold, a drift is detected
                cum_loss_temp = self.cum_loss + loss_batch - self.avg_loss - self.tolerance
                PH = cum_loss_temp - self.mincum_loss
                if PH <= self.thresh:
                    drift = False
            # eliminate the cached data, and adjust the existing parameters
            elim_num += 1
            self.n_batch -= 1
        cached_TS = cached_TS[elim_num * window_size:]
        for key, val in cached_MP.items():
            cached_MP[key] = val[elim_num * window_size:, elim_num * window_size:]
        return cached_TS, cached_MP

    def stream_window(self, dataset, window_size):
        window_size = int(window_size)
        w = dataset[self.t_stamp:self.t_stamp+window_size]
        self.t_stamp += window_size
        return w

