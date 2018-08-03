import os
import numpy as np
import collections
import pickle
import sys
import psutil as ps
import random
import csv
import json
from use.timeseries import TimeSeries


class Utils(object):

    RAW_DIRNAME = "/RawData/"

    SHAPELET_DIRNAME = "/UniShapelets/"
    SHAPELET_EXT = ".shapelet"
    '''
    SEQUENCE_DIRNAME = "/Sequences/"
    SEQUENCE_EXT = ".sequence"
    JSON = SEQUENCE_DIRNAME + "json/"
    '''
    CSV_DIRNAME = "/csv_shapelet/"
    CSV_EXT = ".csv"


    @staticmethod
    def load(directory, option):
        if option.lower() == 'raw data':
            dirname = "/RawData/"
            extension = ".csv"

        files_list = [f for f in os.listdir(directory + dirname) if f.lower().endswith(extension)]
        list_objects = []
        for file in files_list:
            path = directory + dirname + file
            an_object = numpy.genfromtxt(path)
            list_objects.extend(an_object)
        # return a list[Array_TimeSeries]
        return list_objects

    # if the dataset format is changed, just modify this function to adapt it
    @staticmethod
    def generate_timeseries(directory):
        #list_ts = list[TimeSeries]
        dict_ts = {}
        list_rawData = load(directory, 'raw data')
        for d in list_rawData:
            # d[0] is the class of TS in the original data, d[1:] is the data in TS
            t = TimeSeries()
            t.class_timeseries = d[0]
            t.timeseries = d[1:]
            t.name = hash(d[1:])
            dict_ts.extend({t.name:t})
        return dict_ts

    @staticmethod
    def save(directory, list_objects, option):
        if option.lower() == 'shapelet':
            dirname = Utils.SHAPELET_DIRNAME
            extension = Utils.SHAPELET_EXT
        elif option.lower() == 'sequence':
            dirname = Utils.SEQUENCE_DIRNAME
            extension = Utils.SEQUENCE_EXT
        elif option.lower() == 'csv':
            dirname = Utils.CSV_DIRNAME
            extension = Utils.CSV_EXT
        folder = directory + dirname
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            ##clean the historical files
            files_list = [f for f in os.listdir(directory + dirname) if f.lower().endswith(extension)]
            for file in files_list:
                path = directory + dirname + file
                os.remove(path)

        if option.lower() != 'csv':
            for anObject in list_objects:
                file_name = str(anObject.name) + anObject.dimension_name + extension
                path = folder + file_name
                pickle.dump(anObject, open(path, "wb"))
        else:
            file_name = "shapelet_test" + extension
            path = folder + file_name
            i = 0
            with open(path, 'w') as f:
                writer = csv.writer(f, lineterminator='\n', delimiter=';',)
                for anObject in list_objects:
                    i+=1
                    for key in anObject.matching_indices:
                        writer.writerow(["shapelet" + str(i), anObject.name, anObject.dimension_name, anObject.class_shapelet, key, anObject.matching_indices[key], anObject.gain, anObject.subsequence.tolist(),  anObject.min_distance] )


    @staticmethod
    def print_progress(iteration, total, prefix='Progress:', suffix='Complete', decimals=1, barLength=70):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            barLength   - Optional  : character length of bar (Int)
        """
        format_str = "{0:." + str(decimals) + "f}"
        percent = format_str.format(100 * (iteration / float(total)))
        filled_length = int(round(barLength * iteration / float(total)))
        bar = '█' * filled_length + '-' * (barLength - filled_length)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

    @staticmethod
    def check_memory(perc=90):
        mem = ps.virtual_memory()
        if mem.percent >= perc:
            return False
        return True

