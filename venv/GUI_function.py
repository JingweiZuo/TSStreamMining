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
LARGE_FONT= ("Verdana", 12)

class gui_function:
    def __init__(self, master):
        self.filename = 'file name'
        self.training_filename = 'choose training set'
        self.testing_filename = 'choose testing set'
        #transfer the main test part to the class
        self.master = master
        self.dataset = Dataset()
        self.dataset_name = None

    def add_dataset(self):
        self.dataset_name = askopenfilename(parent=self.master, title="Choose a file")
        #l1 = Label(self.master.frame1, text=self.training_filename, font=(10))
        #l1.grid(row=2, sticky=W, columnspan=2)
        array_tsdict = Utils.load_dataset(self.dataset_name)
        dir = self.dataset_name.split("/")
        datasetname = dir[-1]
        self.dataset.update(array_tsdict, datasetname)
        self.master.v_dsname.set(self.dataset.name)
        self.master.v_tslength.set(self.dataset.tslength)
        self.master.v_tsnbr.set(self.dataset.size)
        self.master.v_classnbr.set(len(self.dataset.ClassList))

    def add_training_file(self):
        from tkinter.filedialog import askopenfilename
        #self.master.withdraw()
        self.training_filename = askopenfilename(parent=self.master, title="Choose a file")
        #self.master.v.set(self.training_filename)

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
        DP = sm.mass_v2(data, query)
        # display the figures on the CANVAS of the GUI

        # CANVAS
        self.master.figure.clf()
        a = self.master.figure.add_subplot(111)
        x = range(len(DP))
        a.set_title('Distance Profile at Source position:' + str(index_start))
        a.plot(x, DP, linewidth=0.5)
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
        self.master.figure.clf()
        a = self.master.figure.add_subplot(111)
        x = range(len(MP))
        a.set_title('Matrix Profile of ' + self.nbr_source + ' in ' + self.nbr_target)
        a.plot(x, MP, linewidth=0.5)
        self.master.canvas.show()

    def extractLB(self):
        return 0

    def extractRP(self, master):
        source = master.source
        dp_all, mp_all, self.dist_differ, dist_threshold, dist_side_C, dist_side_nonC = mp.computeDistDiffer(source, master.dataset.tsObjectDir, self.m)
        if str(source.class_timeseries) == str(master.v_class.get()):
            RP = dist_side_C
        else:
            RP = dist_side_nonC
        # CANVAS
        self.master.figure.clf()
        a = self.master.figure.add_subplot(111)
        x = range(len(RP))
        a.set_title('Representative Profile of ' + self.nbr_source + ' in class ' + master.v_class.get())
        a.plot(x, RP, linewidth=0.5)
        self.master.canvas.show()

    def extractDiscP(self, master):
        '''source = master.source
        dp_all, mp_all, dist_differ, dist_threshold, dist_side_C, dist_side_nonC = mp.computeDistDiffer(source, master.dataset.tsObjectDir, self.m)'''
        DiscP = self.dist_differ
        # CANVAS
        self.master.figure.clf()
        a = self.master.figure.add_subplot(111)
        x = range(len(DiscP))
        a.set_title('Discriminative Profile of ' + self.nbr_source + ' in dataset')
        a.plot(x, DiscP, linewidth=0.5)
        self.master.canvas.show()

    def extractShapeletCandidate(self):
        import numpy as np
        import matplotlib.pyplot as plt

        class LineBuilder:
            def __init__(self, line):
                self.line = line
                self.xs = list(line.get_xdata())
                self.ys = list(line.get_ydata())
                self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

            def __call__(self, event):
                print('click', event)
                if event.inaxes != self.line.axes: return
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.line.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()

        self.fig = self.master.figure
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title('click to build line segments')
        line, = self.ax1.plot([0], [0])  # empty line
        linebuilder = LineBuilder(line)
        self.master.canvas.show()

    def extractED(self):
        return 0

    def extractEDMatrix(self):
        return 0

    def predict(self):
        import numpy as np
        import matplotlib.pyplot as plt

        class LineBuilder:
            def __init__(self, line):
                self.line = line
                self.xs = list(line.get_xdata())
                self.ys = list(line.get_ydata())
                self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

            def __call__(self, event):
                print('click', event)
                if event.inaxes != self.line.axes: return
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.line.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()

        self.fig = self.master.figure
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.ax2.set_title('click to build line segments')
        line, = self.ax2.plot([0], [0])  # empty line
        linebuilder = LineBuilder(line)
        self.master.canvas.show()


