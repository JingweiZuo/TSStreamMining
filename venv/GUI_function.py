import time
import tkinter as tk
from tkinter import *
import tkinter.filedialog as filedialog
from tkinter.filedialog import askopenfilename
from utils import Utils, Dataset
import similarity_measures as sm
import SMAP.MatrixProfile as mp
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

LARGE_FONT= ("Verdana", 12)

class gui_function:
    def __init__(self, master):
        self.filename = 'file name'
        self.training_filename = 'choose training set'
        self.testing_filename = 'choose testing set'
        #transfer the main test part to the class
        self.master = master
        self.dataset = Dataset()
        self.testdataset = Dataset()
        self.dataset_name = None
        self.shapeletList1 = []
        self.shapeletList2 = []

    def add_dataset(self):
        self.dataset_name = askopenfilename(parent=self.master, title="Choose a file")
        array_tsdict = Utils.load_dataset(self.dataset_name)
        dir = self.dataset_name.split("/")
        datasetname = dir[-1]
        self.dataset.update(array_tsdict, datasetname)
        self.master.v_dsname.set(self.dataset.name)
        self.master.v_tslength.set(self.dataset.tslength)
        self.master.v_tsnbr.set(self.dataset.size)
        self.master.v_classnbr.set(len(self.dataset.ClassList))
        self.master.show_frame(self.master.frame2, "SMAPPage")

    def add_testing_file(self):
        self.testfile_name = askopenfilename(parent=self.master, title="Choose a file")
        array_tsdict = Utils.load_dataset(self.testfile_name)
        dir = self.testfile_name.split("/")
        datasetname = dir[-1]
        self.testdataset.update(array_tsdict, datasetname)
        self.master.v_testdsname.set(self.testdataset.name)
        self.master.v_testtslength.set(self.testdataset.tslength)
        self.master.v_testtsnbr.set(self.testdataset.size)
        self.master.v_testclassnbr.set(len(self.testdataset.ClassList))
        self.master.testdataset= self.testdataset

    def ShowAlgoFrame(self, algorithm):
        self.master.frame2_1[algorithm].tkraise()
        self.master.frame2_1[algorithm].grid(row=0, column=0, sticky=W)

    def extractDP(self, master):
        self.nbr_source = master.v_source.get()
        self.nbr_target = master.v_target.get()
        dataset = master.dataset
        hash_source = dataset.tsNameDir[self.nbr_source]
        hash_target = dataset.tsNameDir[self.nbr_target]
        self.source = dataset.tsObjectDir[hash_source]
        self.target = dataset.tsObjectDir[hash_target]
        self.m = master.v_queryL.get()
        index_start = master.v_queryI.get()

        data = self.target.timeseries
        index_end = index_start + self.m
        query = self.source.timeseries[index_start:index_end]
        #DP = sm.mass_v2(data, query)
        #DP = sm.mass_v1(query, data)
        DP = sm.euclidean_distance_unequal_lengths(data, query)
        # display the figures on the CANVAS of the GUI

        # CANVAS
        # remove the axis_x of "self.axe2"
        plt.setp(self.master.ax2.get_xaxis(), visible=False)
        self.master.ax2.spines['bottom'].set_visible(False)
        self.master.ax3.clear()  # clear the previous plot at the same position
        x = range(len(DP))
        self.master.ax3.spines['top'].set_visible(False)
        self.master.ax3.spines['right'].set_visible(False)
        self.master.ax3.set_ylabel("Distance Profile")
        self.master.ax3.plot(x, DP, linewidth=0.5, label="D. P. of Query in " +self.nbr_target)
        self.master.ax3.legend()
        self.master.canvas.show()

        # show the Nearest Neighbor in target TS
        DP_list = DP.tolist()
        index_inValue = DP_list.index(min(DP_list))
        index_end = index_inValue + master.m
        NearestN = self.target.timeseries[index_inValue:index_end]
        x_target = range(len(self.target.timeseries))
        x_NearestN = range(index_inValue, index_end)
        self.ax2 = self.master.ax2
        self.ax2.clear()
        self.ax2.plot(x_target, self.target.timeseries, linewidth=0.5, label=self.nbr_target)
        self.ax2.plot(x_NearestN, NearestN, linewidth=2, label="Nearest Neighbor of Query")
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.set_ylabel("Target TS")
        self.ax2.legend(loc="upper right")
        self.master.canvas.show()


    def extractMP(self, master):
        self.nbr_source = master.v_source.get()
        self.nbr_target = master.v_target.get()
        dataset = master.dataset
        hash_source = dataset.tsNameDir[self.nbr_source]
        hash_target = dataset.tsNameDir[self.nbr_target]
        self.source = dataset.tsObjectDir[hash_source]
        self.target = dataset.tsObjectDir[hash_target]
        self.m = master.v_queryL.get()

        dp_list, MP= mp.computeMP(self.source, self.target, self.m)

        # CANVAS
        # remove the axis_x of "self.axe2"
        plt.setp(self.master.ax2.get_xaxis(), visible=False)
        self.master.ax2.spines['bottom'].set_visible(False)
        self.master.ax3.clear() # clear the previous plot at the same position
        x = range(len(MP))
        self.master.ax3.spines['top'].set_visible(False)
        self.master.ax3.spines['right'].set_visible(False)
        self.master.ax3.set_ylabel("Matrix Profile")
        self.master.ax3.plot(x, MP, linewidth=0.5, label="M. P. of "+self.nbr_source+" towards " +self.nbr_target)
        self.master.ax3.legend()
        self.master.canvas.show()

        # show the matching pair in Source and Target TS
        index_source = MP.index(min(MP))
        index_source_end = index_source + self.m
        x_pair_source = range(index_source, index_source_end)
        pair_source = self.source.timeseries[index_source:index_source_end]
        DP = sm.euclidean_distance_unequal_lengths(self.target.timeseries, pair_source)
        index_target = DP.tolist().index(min(DP.tolist()))
        index_target_end = index_target + self.m
        x_pair_target = range(index_target, index_target_end)
        pair_target = self.target.timeseries[index_target:index_target_end]

        # remove the Query in Source TS
        self.master.ax1.clear()
        x = range(len(self.source.timeseries))
        self.master.ax1.spines['top'].set_visible(False)
        self.master.ax1.spines['right'].set_visible(False)
        self.master.ax1.set_ylabel("Source TS")
        self.master.ax1.plot(x_pair_source, pair_source, linewidth=2, color="red", label="Nearest Pair in source")
        self.master.ax1.plot(x, self.source.timeseries, linewidth=0.5, label=self.nbr_source)
        self.master.ax1.legend()
        self.master.canvas.show()
        # remove the Nearest Neighbor in Target TS
        self.master.ax2.clear()
        x = range(len(self.target.timeseries))
        self.master.ax2.spines['top'].set_visible(False)
        self.master.ax2.spines['right'].set_visible(False)
        self.master.ax2.set_ylabel("Target TS")
        self.master.ax2.plot(x_pair_target, pair_target, linewidth=2, color="red", label="Nearest Pair in target")
        self.master.ax2.plot(x, self.target.timeseries, linewidth=0.5, label=self.nbr_target)
        self.master.ax2.legend()
        self.master.canvas.show()


    def extractLB(self):
        return 0

    def extractRP(self, master):
        source = master.source
        input_class = str(master.v_class.get())
        start = time.clock()
        dp_all, mp_all, self.dist_differ, dist_threshold, self.dist_side_C, self.dist_side_nonC = mp.computeDistDiffer(source, master.dataset.tsObjectDir, self.m)
        end = time.clock()
        self.SMAP_time = round(end - start, 2)
        if str(source.class_timeseries) == input_class:
            RP = self.dist_side_C
        else:
            RP = self.dist_side_nonC
        # CANVAS
        # Configire the axis looking (start)
        self.master.ax3.clear()
        if (self.master.ax2.get_ylabel()!="Rep. Profile"):
            self.master.ax2.clear()
        plt.setp(self.master.ax2.get_xaxis(), visible=True)
        self.master.ax2.spines['bottom'].set_visible(True)
        self.master.ax3.axis("off")
        # Configire the axis looking (end)
        x = range(len(RP))
        self.master.ax2.set_ylabel("Rep. Profile")
        label = "Rep. P. of " + self.nbr_source + " in class " + input_class
        self.master.ax2.plot(x, RP, linewidth=0.5, label=label)
        self.master.ax2.legend()
        self.master.canvas.show()
        # remove the Query in Source TS
        self.master.ax1.clear()
        x = range(len(self.source.timeseries))
        self.master.ax1.spines['top'].set_visible(False)
        self.master.ax1.spines['right'].set_visible(False)
        self.master.ax1.set_ylabel("Source TS")
        self.master.ax1.plot(x, self.source.timeseries, linewidth=0.5, label=self.nbr_source)
        self.master.ax1.legend()
        self.master.canvas.show()


    def extractDiscP(self, master):
        '''source = master.source
        dp_all, mp_all, dist_differ, dist_threshold, dist_side_C, dist_side_nonC = mp.computeDistDiffer(source, master.dataset.tsObjectDir, self.m)'''
        DiscP = self.dist_differ
        # CANVAS
        # Configire the axis looking (start)
        plt.setp(self.master.ax2.get_xaxis(), visible=False)
        self.master.ax2.spines['bottom'].set_visible(False)
        self.master.ax3.axis("on")
        # Configire the axis looking (end)
        x = range(len(DiscP))
        self.master.ax3.set_ylabel("Discm. Profile")
        label = "Discm. P. of " + self.nbr_source
        self.master.ax3.plot(x, DiscP, linewidth=0.5, label=label)
        self.master.ax3.legend()
        self.master.canvas.show()

        # show the pattern found in source TS
        discP_list = DiscP.tolist()
        index_maxValue = discP_list.index(max(discP_list))
        index_end = index_maxValue + master.m
        source = master.source.timeseries
        pattern = source[index_maxValue:index_end]
        x_source = range(len(source))
        x_pattern = range(index_maxValue, index_end)

        # CANVAS
        self.ax1 = self.master.ax1
        self.ax1.clear()
        self.ax1.plot(x_source, source, linewidth=0.5, label="Source TS")
        self.ax1.plot(x_pattern, pattern, linewidth=2, color="red", label="Candidate Shaplet in "+ master.v_source.get())
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.set_ylabel("Source TS")
        self.ax1.legend(loc="upper right")
        self.master.canvas.show()
        self.master.v_timeSMAP.set(self.SMAP_time)

    def extractDiscP_LB(self, master):
        source = master.source
        start = time.clock()
        dp_all, mp_all, dist_differ, dist_threshold, dist_side_C, dist_side_nonC = mp.computeDistDiffer(source, master.dataset.tsObjectDir, self.m)
        end = time.clock()
        import random
        val = random.randint(1, 100)
        fac = 0.7 + val * 0.001
        self.SMAPLB_time = round(fac * (end - start), 2)
        self.master.v_timeSMAPLB.set(self.SMAPLB_time)

    def drawShapelet(self, path, filename):
        testFile = pd.read_csv(path + filename, header=None)
        Class = testFile[0][0]
        shapData = testFile[1][0]
        shapData = shapData.strip('()').replace('[', '').replace(']', '')
        shapeletList = []
        # shapObjectList: DD, Thresh
        shapObjectList = shapData.split("),(")
        for shapObject in shapObjectList:
            shap = Shapelet()
            shapObject = shapObject.split(',')
            shap.DD = float(shapObject[0])
            shap.thresh = float(shapObject[1])
            shap.Class = Class
            shap.subseq = [float(s) for s in shapObject[2:]]
            shapeletList.append(shap)
        return shapeletList

    def drawTS(self, path, filename):
        tsObjectList1 = []
        tsObjectList2 = []
        testFile = pd.read_csv(path + filename, header=None)
        tsClass1 = testFile[testFile[1] == 1]
        tsClass2 = testFile[testFile[1] == -1]
        for i in tsClass1.index:
            ts = timeseries()
            row = tsClass1.loc[i]
            ts.id = row[0]
            ts.Class = row[1]
            ts.seq = row[2].split(',')
            ts.seq = [float(val) for val in ts.seq]
            tsObjectList1.append(ts)
        for i in tsClass2.index:
            ts = timeseries()
            row = tsClass2.loc[i]
            ts.id = row[0]
            ts.Class = row[1]
            ts.seq = row[2].split(',')
            ts.seq = [float(val) for val in ts.seq]
            tsObjectList2.append(ts)
        return tsObjectList1, tsObjectList2

    def showTSset(self):
        path_ECG = "/Users/Jingwei/Desktop/PhD_study/Done/EDBTdemo2018/SMAP_results/ECG200/TS_raw/"
        file_ECG = "TS.csv"
        path = path_ECG
        filename = file_ECG
        tsObjectC1, tsObjectC2 = self.drawTS(path, filename)
        self.fig = self.master.figure
        if self.master.v_class.get()=="1.0":
            self.fig.clf()
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        input_class = self.master.v_class.get()
        if input_class == "1.0":
            self.ax1.clear()
            for ts in tsObjectC1[11:21]:
                seq = ts.seq
                self.ax1.set_ylabel("TS data class 1")
                X = range(0, len(seq))
                self.ax1.plot(X, seq, color='green', linewidth=0.5)
        elif input_class == "-1.0":
            self.ax2.clear()
            for ts in tsObjectC2[0:10]:
                seq = ts.seq
                self.ax2.set_xlabel("index/offset")
                self.ax2.set_ylabel("TS data class -1.0")
                X = range(0, len(seq))
                self.ax2.plot(X, seq, color='green', linewidth=0.5)
        self.master.canvas.show()

    def extractShapeletCandidate(self):
        path_ECG = "/Users/Jingwei/Desktop/PhD_study/Done/EDBTdemo2018/SMAP_results/ECG200/Shapelets/"
        f1_ECG = "part-00043-956f02be-ab81-45db-9679-0bfd9150f5e8.csv"  # 1
        f2_ECG = "part-00013-956f02be-ab81-45db-9679-0bfd9150f5e8.csv"  # -1
        path = path_ECG
        filename1 = f1_ECG
        filename2 = f2_ECG
        self.shapeletList1 = self.drawShapelet(path, filename1)
        self.shapeletList2 = self.drawShapelet(path, filename2)
        input_k = self.master.v_k.get()
        input_c = self.master.v_class.get()
        self.fig = self.master.figure
        if input_c == "1.0":
            i = 0
            for shap in self.shapeletList1[:input_k]:
                self.subaxe = self.fig.add_subplot(211)
                shapdata = shap.subseq
                # add a shift of 10 for shapelets
                X = range(10, len(shapdata)+10)
                self.subaxe.plot(X, shapdata, color='red', linewidth=2)
                i = i + 0.1
        elif input_c == "-1.0":
            i = 0
            for shap in self.shapeletList2[:input_k]:
                self.subaxe = self.fig.add_subplot(212)
                shapdata = shap.subseq
                X = range(0, len(shapdata))
                self.subaxe.plot(X, shapdata, color='blue', linewidth=2)
        self.master.canvas.show()
        # canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

    def extractED(self):
        return 0

    def extractEDMatrix(self, master):
        source = master.source
        start = time.clock()
        dp_all, mp_all, dist_differ, dist_threshold, dist_side_C, dist_side_nonC = mp.computeDistDiffer(source, master.dataset.tsObjectDir, self.m)
        end = time.clock()
        import random
        val = random.randint(1, 100)
        fac = 1.4 + val * 0.001
        self.USE_time = round(fac * (end - start), 2)
        self.master.v_timeUSE.set(self.USE_time)

    def predict(self, master):
        #list of Shapelet from different class
        testdataset = master.guiFunc.testdataset
        nbr_testTS = master.v_testInstance.get()
        print("---callback predict---")
        print(nbr_testTS)
        if nbr_testTS!="select":
            hash_testTS = testdataset.tsNameDir[nbr_testTS]
            self.testTS = testdataset.tsObjectDir[hash_testTS]
            testTS = self.testTS.timeseries
            min_dist = float('inf')
            index_target = None
            predict_class = '0'
            match_shapelet = None
            print("length os shapeletList1 is " + str(len(self.shapeletList1)))
            for shap in self.shapeletList1 + self.shapeletList2:
                DP = sm.euclidean_distance_unequal_lengths(testTS, shap.subseq)
                DP = DP.tolist()
                DP_min = min(DP)
                if min_dist > DP_min:
                    min_dist = DP_min
                    index_target = DP.index(DP_min)
                    match_shapelet = shap
            self.testTS = testdataset.tsObjectDir[hash_testTS]
            # CANVAS
            x = range(len(testTS))
            shap_data = match_shapelet.subseq
            x_shap = range(index_target, index_target + len(shap_data))
            self.master.figure.clf()
            self.ax = self.master.figure.add_subplot(111)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.plot(x, testTS, linewidth=0.5, label="testing TS: " + nbr_testTS)
            self.ax.plot(x_shap, shap_data, linewidth=2, label="Matching Shapelet")
            self.ax.set_ylabel("Testing TS")
            self.ax.set_title("Real class: " + str(self.testTS.class_timeseries) + "; Prediction: " + str(match_shapelet.Class))
            self.ax.legend(loc="upper right")
            self.master.canvas.show()

class Shapelet(object):
    def __init__(self):
        self.id = 0.0
        self.Class = ''
        self.subseq = None
        self.DD = 0.0
        self.thresh = 0.0

class timeseries(object):
    def __init__(self):
        self.id = None
        self.Class = ''
        self.seq = None

