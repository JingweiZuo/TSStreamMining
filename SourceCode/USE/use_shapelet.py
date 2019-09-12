from utils.use_utils import old_Utils
import collections
import utils.similarity_measures as sm
import numpy as np


class Shapelet(object):
    ROUNDTO = 3

    def __init__(self):
        self.id = id(self)
        self.name = ''
        # self.time = None
        self.subseq = None
        self.Class = ''
        self.dist_threshold = 0.0
        self.normal_distance = 0.0
        # self.utility_score = 0.0
        # self.row_distances = None
        self.gain = 0.0
        self.dimension_name = ''

        # A dict: for each time series (by name) as a key, it exists a list of floats that represent the distances
        #  between that shapelet and each subsequence of this time series
        self.covering_dict = {}

        # A dict: for each time series (by name) as a key, it exists a float that represent the minimum distance
        # between that shapelet and that time series
        self.min_distance = {}

        # A dict: for each time series (by name) as a key, it exists a list of the indices where this shapelet
        #  matched this time series
        self.matching_indices = {}

    def __repr__(self):
        representation = "{"   # Outermost
        representation += '"id": ' + str(self.id) + ','
        representation += '"name": "' + str(self.name) + '",'
        representation += '"class_shapelet": "' + str(self.Class) + '",'
        representation += '"dist_threshold": ' + str(self.dist_threshold) + ','
        representation += '"gain": ' + str(self.gain) + ','
        representation += '"dimension_name": "' + str(self.dimension_name) + '",'
        representation = old_Utils.json_list(representation, "subsequence", self.subseq)
        representation += ','
        representation = old_Utils.json_dict(representation, 'covering_dict', self.covering_dict)
        representation += ','
        representation = old_Utils.json_dict(representation, 'min_distance', self.min_distance, 'no')
        representation += ','
        representation = old_Utils.json_dict(representation, 'matching_indices', self.matching_indices)
        representation += "}"   # Outermost

        return representation

    def __str__(self):
        representation = "Shapelet with id: " + str(self.id)
        representation += " with class: " + str(self.Class)
        representation += " with distance threshold: " + str(self.dist_threshold)
        representation += " with gain: " + str(self.gain)
        representation += " with dimension: " + self.dimension_name
        return representation

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def check(self, list_timeseries, distance_measure):
        distances_dict = collections.defaultdict(list)
        for a_timeseries in list_timeseries:##O(n)
            distances = sm.calculate_distances(a_timeseries.timeseries, self.subseq, distance_measure)

            min_distance = distances.min()

            ##return min_distance rounded to 3 digits from the decimal point, distance_dict: {distance : a_timeseries}
            distances_dict[round(min_distance, Shapelet.ROUNDTO)].append(a_timeseries)
            #print("distances-dict.keys is : ", distances_dict.keys())
            self.covering_dict[a_timeseries.name] = distances
            self.min_distance[a_timeseries.name] = min_distance
        ordered_distances_dict = collections.OrderedDict(sorted(distances_dict.items()))
        #print("ordered_distances_dict is ", ordered_distances_dict)
        return Shapelet.calculate_information_gain(list_timeseries, ordered_distances_dict)

    def cover(self, list_timeseries):
        cover = False
        remaining = 0
        for a_timeseries in list_timeseries:
            ##By defaut, matched is 'false'
            if not a_timeseries.matched:
                ##if the distance between the shapelet and timeseries < dist_threshold, and one of the timeseries
                if self.min_distance[a_timeseries.name] <= self.dist_threshold:
                    cover = True
                    a_timeseries.matched = True
                else:
                    ##'remaining' is used to verify if the list has been all coverd
                    remaining += 1
        return cover, remaining

    def build_matching_indices(self):
        ## 'distances' is an ArrayList of slices in a 'timeseries', index is the position of each slices
        ## if the index is null, that's to say the timeseries don't contain the shapelet
        for timeseries_name, distances in self.covering_dict.items():
            self.matching_indices[timeseries_name] = [index for index, item in
                                                      enumerate(distances) if item <= self.dist_threshold]

    @staticmethod
    def generate_candidates(list_timeseries, min_length, max_length, skip=False):
        pool = []
        l = max_length
        while l >= min_length:
            if skip and int(l / 4) != 0:
                step = int(l / 4)
            else:
                step = 1
            for aTimeseries in list_timeseries:
                the_name = aTimeseries.name
                the_dimension = aTimeseries.dimension_name
                the_class = aTimeseries.class_timeseries
                for chunk in old_Utils.sliding_window(aTimeseries.timeseries, l, step=step):
                    candidate = Shapelet()
                    candidate.name = the_name
                    candidate.dimension_name = the_dimension
                    candidate.Class = the_class
                    candidate.subseq = chunk
                    pool.append(candidate)
            l -= 1
        return pool

    @staticmethod
    def calculate_information_gain(unid_timeseries, distances_dict):
        #tsclass = [ts.class_timeseries for ts in unid_timeseries]
        #print("tsclass is ", tsclass)
        distances = list(distances_dict.keys())
        gain = 0
        dist_threshold = 0.0
        if len(distances) >= 2:
            ##window size = 2, by defaut, sliding step = 1
            for two_distances in old_Utils.sliding_window(distances, 2):
                two_distances = np.array(two_distances)

                candidate_dist_threshold = two_distances.mean()
                d1 = []
                d2 = []
                ##key is 'distance', value is 'timeseries'
                for key, timeseries in distances_dict.items():
                    ##divide the timeseries list into 2 parts: d1, d2
                    if key < candidate_dist_threshold:
                        d1 += timeseries
                    else:
                        d2 += timeseries
                #print("d1, ", str(d1), "d2, ", str(d2))
                ##calculate the information gain, return the gain maximun and the corresponding 'dist_threshold'
                candidate_gain = old_Utils.dataset_entropy(unid_timeseries) - old_Utils.entropy_after_split(d1, d2)
                '''print("old_Utils.dataset_entropy(unid_timeseries) is ", old_Utils.dataset_entropy(unid_timeseries))
                print("old_Utils.entropy_after_split(d1, d2) is", old_Utils.entropy_after_split(d1, d2))'''
                if candidate_gain > gain:
                    gain = candidate_gain
                    dist_threshold = candidate_dist_threshold

        return gain, dist_threshold

    def get_matching_timeseries(self, list_multivariate_timeseries, index, k=10):
        i = 0
        s = 0
        result = []
        for key in self.matching_indices:
            if i < index:
                i += 1
                continue
            a_timeseries = [t for t in list_multivariate_timeseries if
                            t.name == key and t.class_timeseries == self.Class and self.matching_indices[key]]

            if a_timeseries:
                result.append(a_timeseries[0])
                s += 1
                if s == k:
                    break
        index += s
        if index >= len(list_multivariate_timeseries):
            index = 0
        return result, index + s

    def get_not_matching_timeseries(self, list_multivariate_timeseries, index, k=10):
        i = 0
        s = 0
        result = []
        for aTimeseries in list_multivariate_timeseries:
            if i < index:
                i += 1
                continue
            if aTimeseries.class_timeseries != self.Class:
                result.append(aTimeseries)
                s += 1
                if s == k:
                    break
        index += s
        if index >= len(list_multivariate_timeseries):
            index = 0
        return result, index

    def get_parent_timeseries(self, list_multivariate_timeseries):
        timeseries = [t for t in list_multivariate_timeseries if t.name == self.name]
        index = [i for i, item in enumerate(self.covering_dict[self.name]) if item == 0]
        if not timeseries:
            timeseries = None
        else:
            timeseries = timeseries[0]

        if not index:
            index = None
        else:
            index = index[0]

        return timeseries, index
