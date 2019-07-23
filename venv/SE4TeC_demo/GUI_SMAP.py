import tkinter
from tkinter import *
from tkinter.ttk import *  # Widgets avec thèmes
from SE4TeC_demo.GUI_function import gui_function
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

class SMAPPage(tkinter.Frame):
    # SMAP
    def __init__(self, container, parent):
        self.container = container
        self.parent = parent
        self.dataset = self.parent.guiFunc.dataset
        tkinter.Frame.__init__(self, container)

        # the "textvairiable" of each Combobox:
        self.v_source = StringVar()
        self.v_source.set("select")
        self.v_target = StringVar()
        self.v_queryL = IntVar()
        self.v_queryI = IntVar()
        self.v_class = StringVar()

        self.frame2_1 = tkinter.Frame(self.container, bg="gray90")
        self.frame2_1.grid(row=0, column=0, ipady = 25, sticky="new")
        l_source = tkinter.Label(self.frame2_1, text="Source Time Series :", background="gray90",
                                 foreground="SkyBlue4")

        self.combb_source = Combobox(self.frame2_1, values=self.dataset.tsNbrList, textvariable = self.v_source, state='readonly',
                                                 width=10)
        self.combb_source.bind('<<ComboboxSelected>>', self.callback_source)
        self.m = tkinter.Label(self.frame2_1, text="Query length", bg="gray90", foreground="SkyBlue4")
        # Query length, the index of target TS will change along with the query's length
        self.combb_m = Combobox(self.frame2_1, values=self.dataset.queryLength, textvariable = self.v_queryL, state='readonly', width=5)
        self.combb_m.bind('<<ComboboxSelected>>', self.callback_m)

        l_source.grid(row=0, column=0, sticky=W)
        self.combb_source.grid(row=0, column=1, sticky=W, columnspan=2)
        self.m.grid(row=1, column=0, sticky=W)
        self.combb_m.grid(row=1, column=1, sticky=W)
        # Placement of a single SOURCE/TARGET visualization
        self.frame2_1_1 = tkinter.Frame(self.frame2_1, borderwidth=2, highlightthickness=1, bg="gray90",
                                        highlightbackground="gray90",
                                        highlightcolor="gray90",
                                        relief="sunken")
        self.frame2_1_1.grid(row=2, column=0, sticky=W, columnspan=3, padx=(40, 10), pady=(10, 40))
        l_target = tkinter.Label(self.frame2_1_1, text="Target ", bg="gray90")
        self.combb_target = Combobox(self.frame2_1_1, values=self.dataset.tsNbrList, textvariable = self.v_target, state='readonly', width=10)
        self.combb_target.bind('<<ComboboxSelected>>', self.callback_target)
        l_target.grid(row=0, column=0, sticky=W)
        self.combb_target.grid(row=0, column=1, sticky=W)
        # Placement of a single index for SOURCE/TARGET visualization
        self.frame2_1_2 = tkinter.Frame(self.frame2_1_1, borderwidth=20, bg="gray90")
        self.frame2_1_2.grid(row=1, column=0, sticky=W, columnspan=2)
        self.index = tkinter.Label(self.frame2_1_2, text="index", bg="gray90")
        self.combb_index = Combobox(self.frame2_1_2, values=self.dataset.queryLength, textvariable = self.v_queryI, state='readonly', width=5)
        self.combb_index.bind('<<ComboboxSelected>>', self.callback_index)

        l_DP = tkinter.Label(self.frame2_1_2, text="Distance Profile", bg="gray90")
        b_DP = tkinter.Button(self.frame2_1_2, text="extract", command=lambda: self.parent.guiFunc.extractDP(self), highlightbackground="gray90")
        self.index.grid(row=0, column=0, sticky=W)
        self.combb_index.grid(row=0, column=1, sticky=W)

        l_DP.grid(row=1, column=0, sticky=W)
        b_DP.grid(row=1, column=1, sticky=W)

        # Placement of Matrix Profile for SOURCE/TARGET visualization
        l_MP = tkinter.Label(self.frame2_1_1, text="Matrix Profile", bg="gray90")
        b_MP = tkinter.Button(self.frame2_1_1, text="extract", command=lambda: self.parent.guiFunc.extractMP(self), highlightbackground="gray90",
                              relief='sunken')
        l_MP.grid(row=2, column=0, sticky=W)
        b_MP.grid(row=2, column=1, sticky=W)

        # Placement of Representative/Discrimination Profile for SOURCE visualization
        l_RP = tkinter.Label(self.frame2_1, text="Rep. Profile  in class", bg="gray90", foreground="SkyBlue4")
        combb_class = Combobox(self.frame2_1, values=self.dataset.ClassList, textvariable = self.v_class, state='readonly', width=6)
        b_RP = tkinter.Button(self.frame2_1, text="extract", command=lambda: self.parent.guiFunc.extractRP(self), highlightbackground="gray90")
        l_RP.grid(row=3, column=0, sticky=W)
        combb_class.grid(row=3, column=1, sticky=W + N)
        b_RP.grid(row=3, column=2, sticky=W)
        l_DiscP = tkinter.Label(self.frame2_1, text="Discm. Profile ", bg="gray90", foreground="SkyBlue4")
        b_DiscP = tkinter.Button(self.frame2_1, text="extract", command=lambda: self.parent.guiFunc.extractDiscP(self), highlightbackground="gray90")
        b_USE = tkinter.Button(self.frame2_1, text="time(USE)",
                                  command=lambda: self.parent.guiFunc.extractEDMatrix(self),
                                  highlightbackground="gray90")
        b_SMAPLB = tkinter.Button(self.frame2_1, text="speedup(LB)", command=lambda: self.parent.guiFunc.extractDiscP_LB(self),
                                 highlightbackground="gray90")
        l_DiscP.grid(row=4, column=0, sticky=W)
        b_DiscP.grid(row=4, column=1, sticky=E, columnspan=2)
        b_USE.grid(row=5, column=0, sticky=E, columnspan=1)
        b_SMAPLB.grid(row=5, column=1, sticky=E, columnspan=2)
    def callback_m(self, event=None):
        print('--- callback_m ---')
        queryL = self.v_queryL.get()
        RealIndex = list(range(self.dataset.tslength - int(queryL)))
        self.combb_index.configure(values=RealIndex)

    def callback_source(self, event=None):
        print('--- callback_source ---')
        dataset = self.dataset
        nbr_source = self.v_source.get()
        hash_source = dataset.tsNameDir[nbr_source]
        self.source = dataset.tsObjectDir[hash_source]
        self.parent.v_sourceC.set(self.source.class_timeseries)
        # CANVAS
        source = self.source.timeseries
        x = range(len(source))
        self.ax1 = self.parent.ax1
        self.ax1.clear()  # clear the previous plot at the same position
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.plot(x, source, linewidth=0.5, label=nbr_source)
        self.ax1.set_ylabel("Source TS")
        self.ax1.legend(loc="upper right")
        self.parent.canvas.show()

    def callback_target(self, event=None):
        print('--- callback_target ---')
        dataset = self.dataset
        nbr_target = self.v_target.get()
        hash_target = dataset.tsNameDir[nbr_target]
        self.target = dataset.tsObjectDir[hash_target]
        self.parent.v_targetC.set(self.target.class_timeseries)
        # CANVAS
        # remove the axis_x of "self.axe1"
        plt.setp(self.ax1.get_xaxis(), visible=False)
        self.ax1.spines['bottom'].set_visible(False)
        target = self.target.timeseries
        x = range(len(target))
        self.ax2 = self.parent.ax2
        self.ax2.clear()  # clear the previous plot at the same position
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.plot(x, target, linewidth=0.5, label=nbr_target)
        self.ax2.set_ylabel("Target TS")
        self.ax2.legend(loc="upper right")
        self.parent.canvas.show()

    def callback_index(self, event=None):
        print('--- callback_index ---')
        dataset = self.dataset
        self.m = self.v_queryL.get()
        index_start = self.v_queryI.get()
        index_end = index_start + self.m
        x_query = range(index_start,index_end)
        self.query = self.source.timeseries[index_start:index_end]
        # CANVAS
        self.ax1 = self.parent.ax1
        self.ax1.clear()
        source = self.source.timeseries
        x_source = range(len(source))
        self.ax1.plot(x_source, source, linewidth=0.5, label=self.v_source.get())
        self.ax1.plot(x_query, self.query, linewidth=2, label="Query")
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.set_ylabel("Source TS")
        self.ax1.legend(loc="upper right")
        self.parent.canvas.show()