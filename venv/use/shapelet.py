import use.similarity_measures as sm
import numpy as np

class Shapelet(object):
    def __init__(self):
        self.id = id(self)
        self.name = ''
        # self.time = None
        self.subsequence = None
        self.class_shapelet = ''
        self.differ_distance = 0.0
        self.dist_threshold = 0.0
        self.dimension_name = ''

        # A dict: for each time series (by name) as a key, it exists a list of the indices where this shapelet
        #  matched this time series
        self.matching_ts = []
        self.matching_indices = {}

