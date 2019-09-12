import tkinter
from tkinter import *
from tkinter.ttk import *  #Widgets avec thèmes
from SE4TeC_demo.GUI_function import gui_function
from SE4TeC_demo.GUI_USE import USEPage
from SE4TeC_demo.GUI_SMAP import SMAPPage
from SE4TeC_demo.GUI_SMAPLB import SMAPLBPage
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageTk
LARGE_FONT= ("Verdana", 12)

from matplotlib import pyplot as plt
import numpy as np

'''class MyProgramme(Tk):
    def __init__(self, *args, **kwargs):

        Tk.__init__(self, *args, **kwargs)
        container = tkinter.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        self.frames = {}
        '''
'''
        for F in (StartPage, PageOne):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)
        '''
'''
        frame = StartPage(container,self)
        self.frames[StartPage] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()'''

class StartPage(Tk):
    def hello(self):
        print("hello cmd")

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        #Pre-configuration of interface
        self.winfo_toplevel().title("SE4TeC - Shapelets Extraction")
        self._root().geometry('820x620')
        self.k_list = list(range(10))
        self.guiFunc = gui_function(self)

        '''start of menubar'''
        menubar = Menu(self)
        filemenu = Menu(menubar, tearoff=0)

        filemenu.add_command(label ="New training file", command = self.guiFunc.add_dataset)
        # add the tkinter.Label which indicate the file name
        filemenu.add_separator()
        filemenu.add_command(label ="exit", command = self.hello)
        menubar.add_cascade(label="File", menu = filemenu)
        self._root().config(menu=menubar)
        '''end of menubar'''

        #Frame1
        self.frame1 = tkinter.Frame(self, bg= "SkyBlue4")
        #self.frame1.pack(fill = BOTH, side=TOP)
        self.frame1.pack(fill=X)
        self.frame1.config(borderwidth=2, relief=GROOVE)

        #Frame 1_1
        self.frame1_1 = tkinter.Frame(self.frame1, bg= "SkyBlue4")
        self.frame1_1.grid(row=0, column=0, sticky=W)
        l1 = tkinter.Label(self.frame1_1, text="SE4TeC", background= "SkyBlue4", font = "Helvetica 28 bold", foreground = "White")
        l1.grid(row=0, column=0, sticky=W, rowspan=2)

        self.frame1_2 = tkinter.Frame(self.frame1, bg= "SkyBlue4")
        self.frame1_2.grid(row=0, column=1, sticky=W)
        '''b1 = tkinter.Button(self.frame1_2, text="Import Dataset", command=self.guiFunc.add_dataset, highlightthickness=4,
                      #highlightcolor="SkyBlue4",
                      anchor="center",
                      highlightbackground="SkyBlue4",
                      borderwidth=4)'''
        b1 = tkinter.Button(self.frame1_2, text="Import Dataset", command=self.guiFunc.add_dataset,
                            highlightthickness=4,
                            # highlightcolor="SkyBlue4",
                            anchor="center",
                            highlightbackground="SkyBlue4",
                            borderwidth=4)
        b1.grid(row=0, column=0, sticky=W, padx=15, pady=10)

        #Frame 2: the middle part of GUI
        self.frame2 = tkinter.Frame(self, bg= "gray90")
        self.frame2.pack(fill=BOTH)
        self.frame2.config(borderwidth=2, relief=GROOVE)

        # Placement of a single SOURCE visualization
        self.frame2_1 = SMAPPage(self.frame2, self)

        #Frame 1_3
        self.frame1_3 = tkinter.Frame(self.frame1, bg= "SkyBlue4")
        self.frame1_3.grid(row=0, column =2, sticky=E, columnspan=2,padx=80)
        self.img_open = Image.open('CombinedLogo_360_40.jpg')
        self.img = ImageTk.PhotoImage(self.img_open)
        self.label_img = tkinter.Label(self.frame1_3, image=self.img)
        self.label_img.pack()

        '''l_algo = tkinter.Label(self.frame1_3, text="Select the algorithm: ", background= "SkyBlue4", foreground = "White")
        radVar = StringVar()
        b_USE = tkinter.Radiobutton(self.frame1_3, text="USE", variable=radVar, value="USEPage", indicatoron=0, command=lambda x=radVar: self.show_frame(self.frame2, x.get()), border=4)
        b_SMAP = tkinter.Radiobutton(self.frame1_3, text="SMAP", variable=radVar, value="SMAPPage", indicatoron=0, command=lambda x=radVar: self.show_frame(self.frame2, x.get()), border=4)
        b_SMAP_LB = tkinter.Radiobutton(self.frame1_3, text="SMAP_LB", variable=radVar, value="SMAPLBPage", indicatoron=0, command=lambda x=radVar: self.show_frame(self.frame2, x.get()), border=4)
        l_algo.grid(row=0, column=0, sticky=E, padx=(15,5), pady=10)  # position "West"
        b_USE.grid(row=0, column=1, ipadx=15, ipady=10, padx=15, pady=10)
        b_SMAP.grid(row=0, column=2, ipadx=15, ipady=10, padx=15, pady=10)
        b_SMAP_LB.grid(row=0, column=3, ipadx=15, ipady=10, padx=15, pady=10)'''
        self.frame1_3.grid_columnconfigure(0, weight=1)

        ####################################################################################
        #############################Frame 2_2, plot of results#############################
        self.frame2_2 = tkinter.Frame(self.frame2)
        self.frame2_2.grid(row=0, column=1, sticky=N, columnspan=2)
        self.figure = plt.figure(figsize=(8,6), dpi=60)
        self.ax1 = self.figure.add_subplot(311)
        self.ax2 = self.figure.add_subplot(312, sharex = self.ax1)
        self.ax3 = self.figure.add_subplot(313, sharex = self.ax1)
        self.ax1.axis("off")
        self.ax2.axis("off")
        self.ax3.axis("off")
        self.canvas = FigureCanvasTkAgg(self.figure, self.frame2_2)
        self.canvas.show()
        #canvas.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=E)

        toolbar = NavigationToolbar2Tk(self.canvas, self.frame2_2)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        #############################Frame 2_2, plot of results#############################
        ####################################################################################

        ####################################################################################
        #########################Frame 2_3, Properties of Training Dataset##################
        self.frame2_3 = tkinter.Frame(self.frame2, bg = "gray90")
        self.frame2_3.grid(row = 1, column = 0, sticky="wn")
        l_info = tkinter.Label(self.frame2_3, text="Info. of Training Dataset: ", background="SkyBlue4", foreground="White")

        l_info.grid(row = 0, column = 0, sticky="ws", columnspan=2)   # position "West"
        #self.frame2.rowconfigure(0, weight=3)
        #blue
        l_info1 = Label(self.frame2_3, text="TS length: ", foreground="blue")
        l_info2 = Label(self.frame2_3, text="TS number: ", foreground="blue")
        l_info3 = Label(self.frame2_3, text="Nbr class: ", foreground="blue")

        l_info4 = Label(self.frame2_3, text="Source class: ", foreground="blue")
        l_info5 = Label(self.frame2_3, text="Target class: ", foreground="blue")

        self.v_dsname = StringVar()
        self.v_tslength = IntVar()
        self.v_tsnbr = IntVar()
        self.v_classnbr = IntVar()
        self.v_sourceC = StringVar()
        self.v_targetC = StringVar()

        l_dataset = Label(self.frame2_3, textvariable=self.v_dsname, foreground="red")
        l_tslength = Label(self.frame2_3, textvariable=self.v_tslength, foreground="red")
        l_tsnbr = Label(self.frame2_3, textvariable=self.v_tsnbr, foreground="red")
        l_classnbr = Label(self.frame2_3, textvariable=self.v_classnbr, foreground="red")

        l_sourceC = Label(self.frame2_3, textvariable=self.v_sourceC, foreground="red")
        l_targetC = Label(self.frame2_3, textvariable=self.v_targetC, foreground="red")
        l_dataset.grid(row = 1, column = 0, sticky="ws", columnspan=2)  # position "West"
        l_info1.grid(row=2, column=0, sticky="ws")  # position "West"
        l_info2.grid(row=3, column=0, sticky="ws")  # position "West"
        l_info3.grid(row=4, column=0, sticky="ws")  # position "West"
        l_tslength.grid(row=2, column=1, sticky="ws")  # position "West"
        l_tsnbr.grid(row=3, column=1, sticky="ws")  # position "West"
        l_classnbr.grid(row=4, column=1, sticky="ws")  # position "West"
        l_tsnbr.grid(row=3, column=1, sticky="ws")  # position "West"
        l_classnbr.grid(row=4, column=1, sticky="ws")  # position "West"

        l_info4.grid(row=2, column=2, sticky="ws")  # position "West"
        l_info5.grid(row=3, column=2, sticky="ws")  # position "West"
        l_sourceC.grid(row=2, column=3, sticky="ws")  # position "West"
        l_targetC.grid(row=3, column=3, sticky="ws")  # position "West"
        #########################Frame 2_3, Properties of Training Dataset##################
        ####################################################################################

        ####################################################################################
        #########################Frame 2_3, Properties of Testing Dataset###################
        self.frame2_4 = tkinter.Frame(self.frame2, bg="gray90")
        self.frame2_4.grid(row=1, column=1, sticky="wn")
        l_testinfo = tkinter.Label(self.frame2_4, text="Info. of Testing Dataset: ", background="SkyBlue4",
                               foreground="White")

        l_testinfo.grid(row=0, column=0, sticky="ws", columnspan=2)  # position "West"
        # self.frame2.rowconfigure(0, weight=3)
        # blue
        l_testinfo1 = Label(self.frame2_4, text="TS length: ", foreground="blue")
        l_testinfo2 = Label(self.frame2_4, text="TS number: ", foreground="blue")
        l_testinfo3 = Label(self.frame2_4, text="Nbr class: ", foreground="blue")

        self.v_testdsname = StringVar()
        self.v_testtslength = IntVar()
        self.v_testtsnbr = IntVar()
        self.v_testclassnbr = IntVar()

        l_testdataset = Label(self.frame2_4, textvariable=self.v_testdsname, foreground="red")
        l_testtslength = Label(self.frame2_4, textvariable=self.v_testtslength, foreground="red")
        l_testtsnbr = Label(self.frame2_4, textvariable=self.v_testtsnbr, foreground="red")
        l_testclassnbr = Label(self.frame2_4, textvariable=self.v_testclassnbr, foreground="red")

        l_testdataset.grid(row=1, column=0, sticky="ws", columnspan=2)  # position "West"
        l_testinfo1.grid(row=2, column=0, sticky="ws")  # position "West"
        l_testinfo2.grid(row=3, column=0, sticky="ws")  # position "West"
        l_testinfo3.grid(row=4, column=0, sticky="ws")  # position "West"
        l_testtslength.grid(row=2, column=1, sticky="ws")  # position "West"
        l_testtsnbr.grid(row=3, column=1, sticky="ws")  # position "West"
        l_testclassnbr.grid(row=4, column=1, sticky="ws")  # position "West"
        #########################Frame 2_4, Properties of Testing Dataset###################
        ####################################################################################

        ####################################################################################
        #########################Frame 2_4_1, Properties of Testing Dataset###################
        self.frame2_4_1 = tkinter.Frame(self.frame2, bg="gray90")
        self.frame2_4_1.grid(row=1, column=2, sticky="wn")
        l_timeinfo = tkinter.Label(self.frame2_4_1, text="Running Time (s): ", background="SkyBlue4",
                                  foreground="White")
        l_timeinfo.grid(row=0, column=0, sticky="ws", columnspan=2)  # position "West"
        l_timeinfo1 = Label(self.frame2_4_1, text="USE ", foreground="blue")
        l_timeinfo2 = Label(self.frame2_4_1, text="SMAP: ", foreground="blue")
        l_timeinfo3 = Label(self.frame2_4_1, text="SMAP_LB: ", foreground="blue")
        self.v_timeUSE = IntVar()
        self.v_timeSMAP = IntVar()
        self.v_timeSMAPLB = IntVar()
        l_timeUSE = Label(self.frame2_4_1, textvariable=self.v_timeUSE, foreground="red")
        l_timeSMAP = Label(self.frame2_4_1, textvariable=self.v_timeSMAP, foreground="red")
        l_timeSMAPLB = Label(self.frame2_4_1, textvariable=self.v_timeSMAPLB, foreground="red")
        l_timeinfo1.grid(row=1, column=0, sticky="ws")  # position "West"
        l_timeinfo2.grid(row=2, column=0, sticky="ws")  # position "West"
        l_timeinfo3.grid(row=3, column=0, sticky="ws")  # position "West"
        l_timeUSE.grid(row=1, column=1, sticky="ws")  # position "West"
        l_timeSMAP.grid(row=2, column=1, sticky="ws")  # position "West"
        l_timeSMAPLB.grid(row=3, column=1, sticky="ws")  # position "West"
        #########################Frame 2_4_1, Properties of Testing Dataset###################
        ####################################################################################

        # Frame 3, extract Shapelets, predict the test instance
        self.frame3 = tkinter.Frame(self, bg = "gray90")
        self.frame3.pack(fill=X)
        self.frame3.config(borderwidth=2, relief=GROOVE)


        ######################################################################################
        #########################Frame3_1, Show raw TS & Shapelets############################
        self.frame3_1 = tkinter.Frame(self.frame3, bg= "gray90")
        self.frame3_1.grid(row = 0, column = 0, sticky = W )

        self.v_class = StringVar()
        self.v_k = IntVar()
        l_class = tkinter.Label(self.frame3_1, text="Class",bg="gray90")
        combb_shap_class = Combobox(self.frame3_1, values=["1.0", "-1.0"], textvariable = self.v_class , state='readonly', width=6)
        b_show_ds = tkinter.Button(self.frame3_1, text="show TS set", command=self.guiFunc.showTSset, highlightthickness=4,
                      anchor="center",
                      highlightbackground="gray90",
                      borderwidth=0)
        l_k = tkinter.Label(self.frame3_1, text="select k", bg= "gray90")
        combb_k = Combobox(self.frame3_1, values=self.k_list, textvariable = self.v_k, state='readonly', width=5)
        b_shap_extract = tkinter.Button(self.frame3_1, text="Show Shapelets", command=self.guiFunc.extractShapeletCandidate, highlightthickness=4,
                      anchor="center",
                      highlightbackground="gray90",
                      borderwidth=0)
        l_class.grid(row=0, column=0, sticky=W)
        combb_shap_class.grid(row=0, column=1, sticky=W)
        b_show_ds.grid(row = 1, column = 0, sticky = W, columnspan=2)

        l_k.grid(row = 0, column = 2, sticky = W )
        combb_k.grid(row = 0, column = 3, sticky = W )
        b_shap_extract.grid(row = 1, column = 2, sticky = W+E+N+S, columnspan=2)
        #########################Frame3_1, Show raw TS & Shapelets############################
        ######################################################################################

        self.frame3_2 = tkinter.Frame(self.frame3, bg= "gray90")
        self.frame3_2.grid(row = 0, column = 2, sticky = E, padx=(100,0) )
        self.v_testInstance = StringVar()
        self.v_testInstance.set("select")
        b_testfile = tkinter.Button(self.frame3_2, text="Import testing data", command=self.guiFunc.add_testing_file, highlightthickness=4,
                                   # highlightcolor="SkyBlue4",
                                   anchor="center",
                                   highlightbackground="gray90",
                                   borderwidth=0)
        self.combb_test = Combobox(self.frame3_2, values=self.guiFunc.testdataset.tsNbrList, textvariable=self.v_testInstance, postcommand=self.update_combbtest,state='readonly', width=10)
        self.combb_test.bind('<<ComboboxSelected>>', self.callback_test)
        b_predict = tkinter.Button(self.frame3_2, text="predict", command=lambda :self.guiFunc.predict(self), highlightthickness=4,
                      #highlightcolor="SkyBlue4",
                      anchor="center",
                      highlightbackground="gray90",
                      borderwidth=0)
        b_testfile.grid(row = 0, column = 0, sticky = W )
        self.combb_test.grid(row = 0, column = 1, sticky = W )
        b_predict.grid(row = 1, column = 0, sticky = W+E+N+S, columnspan = 2)

    def update_combbtest(self):
        self.combb_test['values']=self.guiFunc.testdataset.tsNbrList

    def show_frame(self, StartPageControl, algo):
        if algo == "USEPage":
            frame = USEPage(StartPageControl, self)
        elif algo == "SMAPPage":
            frame = SMAPPage(StartPageControl, self)
        elif algo == "SMAPLBPage":
            frame = SMAPLBPage(StartPageControl, self)
        #frame.grid(row=0, column=0, sticky=W)
        frame.tkraise()

    def callback_class(self):
        return 0

    def callback_test(self, event=None):
        print('--- callback_test ---')
        testdataset = self.guiFunc.testdataset
        nbr_testTS = self.v_testInstance.get()
        hash_testTS = testdataset.tsNameDir[nbr_testTS]
        self.testTS = testdataset.tsObjectDir[hash_testTS]
        # CANVAS
        testTS = self.testTS.timeseries
        x = range(len(testTS))
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.plot(x, testTS, linewidth=0.5, label="testing TS: " + nbr_testTS)
        self.ax.set_ylabel("Testing TS")
        self.ax.set_title("Real class: " + str(self.testTS.class_timeseries))
        self.ax.legend(loc="upper right")
        self.canvas.show()

if __name__ == '__main__':
    application = StartPage()
    application.mainloop()

'''
    # <Button-1>: event produced by the left tkinter.Button of mouse
    #frame.bind("<ButtonRelease-1>", popup)
'''


