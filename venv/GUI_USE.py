import tkinter
from tkinter import *
from tkinter.ttk import *  # Widgets avec thèmes
from GUI_function import gui_function
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

class USEPage(tkinter.Frame):
    # USE
    def __init__(self, container, master):
        self.container = container
        self.master = master
        self.dataset = self.master.guiFunc.dataset
        tkinter.Frame.__init__(self, container)

        # the "textvairiable" of each Combobox:
        self.v_source = StringVar()
        self.v_source.set("select")
        self.v_target = StringVar()
        self.v_queryL = StringVar()
        self.v_queryI = StringVar()
        self.v_k = StringVar()

        self.frame2_1 = tkinter.Frame(self.container, bg="gray90")
        self.frame2_1.grid(row=0, column=0, ipady = 50, sticky = "new")
        l_source = tkinter.Label(self.frame2_1, text="Source Time Series:", background="gray90",
                                 foreground="SkyBlue4")

        self.combb_source = tkinter.ttk.Combobox(self.frame2_1, values=self.dataset.tsNbrList, textvariable = self.v_source, state='readonly', width=10)
        self.m = tkinter.Label(self.frame2_1, text="Query length", bg="gray90", foreground="SkyBlue4")
        #Query length, the index of target TS will change along with the query's length
        self.combb_m = Combobox(self.frame2_1, values=self.dataset.queryLength, textvariable = self.v_queryL, state='readonly', width=5)
        self.combb_m.bind('<<ComboboxSelected>>', self.callback)

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
        l_target = tkinter.Label(self.frame2_1_1, text="Target", bg="gray90")
        self.combb_target = Combobox(self.frame2_1_1, values=self.dataset.tsNbrList, textvariable = self.v_target, state='readonly', width=10)
        l_target.grid(row=0, column=0, sticky=W)
        self.combb_target.grid(row=0, column=1, sticky=W)
        # Placement of a single index for SOURCE/TARGET visualization
        self.frame2_1_2 = tkinter.Frame(self.frame2_1_1, borderwidth=20, bg="gray90")
        self.frame2_1_2.grid(row=1, column=0, sticky=W, columnspan=2)
        self.index = tkinter.Label(self.frame2_1_2, text="index", bg="gray90")
        self.combb_index = Combobox(self.frame2_1_2, values=self.dataset.queryLength, textvariable = self.v_queryI, state='readonly', width=5)

        l_ED = tkinter.Label(self.frame2_1_2, text="Euclidean Distance", bg="gray90")
        b_ED = tkinter.Button(self.frame2_1_2, text="extract", command=gui_function.extractED, highlightbackground="gray90")
        self.index.grid(row=0, column=0, sticky=W)
        self.combb_index.grid(row=0, column=1, sticky=W)

        l_ED.grid(row=1, column=0, sticky=W)
        b_ED.grid(row=1, column=1, sticky=W)

        # Placement of Matrix Profile for SOURCE/TARGET visualization
        l_EDMatrix = tkinter.Label(self.frame2_1_1, text="Full Distance (S-T)", bg="gray90")
        b_EDMatrix = tkinter.Button(self.frame2_1_1, text="extract", command=gui_function.extractEDMatrix, highlightbackground="gray90",
                              relief='sunken')
        l_EDMatrix.grid(row=2, column=0, sticky=W)
        b_EDMatrix.grid(row=2, column=1, sticky=W)

        # Placement of Representative/Discrimination Profile for SOURCE visualization
        l_patternK = tkinter.Label(self.frame2_1, text="Top-k Pattern", bg="gray90", foreground="SkyBlue4")
        combb_k = Combobox(self.frame2_1, values=[1, 5, 10], textvariable = self.v_k, state='readonly', width=6)
        b_patternK = tkinter.Button(self.frame2_1, text="extract", command=gui_function.extractShapeletCandidate, highlightbackground="gray90")
        l_patternK.grid(row=3, column=0, sticky=W)
        combb_k.grid(row=3, column=1, sticky=W + N)
        b_patternK.grid(row=3, column=2, sticky=W)

    def callback(self, event=None):
        print('--- callback ---')
        queryL = self.v_queryL.get()
        RealIndex = list(range(self.dataset.tslength - int(queryL)))
        self.combb_index.configure(values=RealIndex)