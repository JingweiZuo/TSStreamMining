import numpy as np
from collections import defaultdict
import sys
from typing import Any


class TimeSeries(object):
    def __init__(self):
        self.class_timeseries = ''
        self.dimension_name = ''
        self.timeseries = None
        self.matched = False
        self.name = ''

    def __repr__(self):
        representation = "Timeseries with dimension: " + self.dimension_name
        representation += " with class: " + self.class_timeseries
        representation += " with series: " + str(self.timeseries)
        return representation

    def __str__(self):
        representation = "Timeseries with dimension: " + self.dimension_name
        representation += " with class: " + self.class_timeseries
        representation += " with series: " + str(self.timeseries)
        return representation

    @staticmethod
    def groupByClass_timeseries(list_timeseries):
        dict_ts = {}  # type: dict
        for ts in list_timeseries:
            dict_ts[ts.class_timeseries].append(ts)
        return dict_ts

    @staticmethod
    def generate_timeseries(unid):
        unid_timeseries = []
        for cuts in unid:
            target_class = cuts[-1]
            ts_str = cuts[:-1]
            ts = [float(element) for element in ts_str]
            timeseries = TimeSeries()
            timeseries.class_timeseries = target_class
            timeseries.timeseries = np.array(ts)
            unid_timeseries.append(timeseries)
        return unid_timeseries


