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
        self.normal_distance = 0.0
        self.dist_threshold = 0.0
        self.dimension_name = ''

        # [ts_target_name1, ts_target_name2, ...], Array[string]
        self.matching_ts = []
        # {ts_target_name:[idx1,idx2,...]}, dict{String:Array[]}
        self.matching_indices = {}

