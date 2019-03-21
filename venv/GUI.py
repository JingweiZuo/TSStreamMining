import tkinter
from tkinter import *
from tkinter.ttk import *  #Widgets avec thèmes
from GUI_function import gui_function
from GUI_USE import USEPage
from GUI_SMAP import SMAPPage
from GUI_SMAPLB import SMAPLBPage
from ml_methods import ml_methodes
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
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
        self._root().geometry('805x620')
        self.k_list = [5, 10, 20]
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
        self.frame2_1 = USEPage(self.frame2, self)

        #Frame 1_3
        self.frame1_3 = tkinter.Frame(self.frame1, bg= "SkyBlue4")
        self.frame1_3.grid(row=0, column =2, sticky=W)
        l_algo = tkinter.Label(self.frame1_3, text="Select the algorithm: ", background= "SkyBlue4", foreground = "White")

        radVar = StringVar()
        b_USE = tkinter.Radiobutton(self.frame1_3, text="USE", variable=radVar, value="USEPage", indicatoron=0, command=lambda x=radVar: self.show_frame(self.frame2, x.get()), border=4)
        b_SMAP = tkinter.Radiobutton(self.frame1_3, text="SMAP", variable=radVar, value="SMAPPage", indicatoron=0, command=lambda x=radVar: self.show_frame(self.frame2, x.get()), border=4)
        b_SMAP_LB = tkinter.Radiobutton(self.frame1_3, text="SMAP_LB", variable=radVar, value="SMAPLBPage", indicatoron=0, command=lambda x=radVar: self.show_frame(self.frame2, x.get()), border=4)

        l_algo.grid(row=0, column=0, sticky=E, padx=(15,5), pady=10)  # position "West"
        b_USE.grid(row=0, column=1, ipadx=15, ipady=10, padx=15, pady=10)
        b_SMAP.grid(row=0, column=2, ipadx=15, ipady=10, padx=15, pady=10)
        b_SMAP_LB.grid(row=0, column=3, ipadx=15, ipady=10, padx=15, pady=10)
        self.frame1_3.grid_columnconfigure(0, weight=1)

        #############################Frame 2_2, plot of results#############################
        ####################################################################################
        self.frame2_2 = tkinter.Frame(self.frame2)
        self.frame2_2.grid(row=0, column=1, sticky=E)
        self.figure = Figure(figsize=(5,4), dpi=80)

        self.canvas = FigureCanvasTkAgg(self.figure, self.frame2_2)
        self.canvas.show()
        #canvas.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=E)

        #toolbar = NavigationToolbar2Tk(canvas, self.frame2_2)
        #toolbar.update()
        #canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        #############################Frame 2_2, plot of results#############################
        ####################################################################################

        # Frame 2_3, Properties of Dataset
        self.frame2_3 = tkinter.Frame(self.frame2, bg = "gray90")
        self.frame2_3.grid(row = 1, column = 0, sticky="ws")
        l_info = tkinter.Label(self.frame2_3, text="Information of Dataset: ", background="SkyBlue4", foreground="White")

        l_info.grid(row = 0, column = 0, sticky="ws", columnspan=2)   # position "West"
        #self.frame2.rowconfigure(0, weight=3)
        #blue
        l_info1 = Label(self.frame2_3, text="dataset: ", foreground="blue")
        l_info2 = Label(self.frame2_3, text="TS length: ", foreground="blue")
        l_info3 = Label(self.frame2_3, text="TS number: ", foreground="blue")
        l_info4 = Label(self.frame2_3, text="Nbr class: ", foreground="blue")

        l_info5 = Label(self.frame2_3, text="Source class: ", foreground="blue")
        l_info6 = Label(self.frame2_3, text="Target class: ", foreground="blue")

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

        l_info1.grid(row=1, column=0, sticky="ws")  # position "West"
        l_info2.grid(row=2, column=0, sticky="ws")  # position "West"
        l_info3.grid(row=3, column=0, sticky="ws")  # position "West"
        l_info4.grid(row=4, column=0, sticky="ws")  # position "West"
        l_dataset.grid(row = 1, column = 1, sticky="ws")  # position "West"
        l_tslength.grid(row=2, column=1, sticky="ws")  # position "West"
        l_tsnbr.grid(row=3, column=1, sticky="ws")  # position "West"
        l_classnbr.grid(row=4, column=1, sticky="ws")  # position "West"
        l_tsnbr.grid(row=3, column=1, sticky="ws")  # position "West"
        l_classnbr.grid(row=4, column=1, sticky="ws")  # position "West"

        l_info5.grid(row=1, column=2, sticky="ws")  # position "West"
        l_info6.grid(row=2, column=2, sticky="ws")  # position "West"
        l_sourceC.grid(row=1, column=3, sticky="ws")  # position "West"
        l_targetC.grid(row=2, column=3, sticky="ws")  # position "West"


        '''
        "nbr of instance"
        "nbr of class"
        "length of instance"
'''

        # Frame 3, extract Shapelets, predict the test instance
        self.frame3 = tkinter.Frame(self, bg = "gray90")
        self.frame3.pack(fill=X)
        self.frame3.config(borderwidth=2, relief=GROOVE)

        self.frame3_1 = tkinter.Frame(self.frame3, bg= "gray90")
        self.frame3_1.grid(row = 0, column = 0, sticky = W )
        l_k = tkinter.Label(self.frame3_1, text="select k", bg= "gray90")
        combb_k = Combobox(self.frame3_1, values=self.k_list, state='readonly', width=5)
        #combb_k.config(highlightthickness=0)
        l_shap_class = tkinter.Label(self.frame3_1, text="select the class", bg= "gray90")
        combb_shap_class = Combobox(self.frame3_1, values=self.guiFunc.dataset.ClassList, state='readonly', width=6)
        b_shap_extract = tkinter.Button(self.frame3_1, text="extract the shapelets", command=self.guiFunc.extractShapeletCandidate, highlightthickness=4,
                      #highlightcolor="SkyBlue4",
                      anchor="center",
                      highlightbackground="gray90",
                      borderwidth=0)
        l_k.grid(row = 0, column = 0, sticky = W )
        combb_k.grid(row = 0, column = 1, sticky = W )
        l_shap_class.grid(row = 0, column = 2, sticky = W )
        combb_shap_class.grid(row = 0, column = 3, sticky = W )
        b_shap_extract.grid(row = 1, column = 1, sticky = W+E+N+S, columnspan=2)

        self.frame3_2 = tkinter.Frame(self.frame3, bg= "gray90")
        self.frame3_2.grid(row = 0, column = 2, sticky = E, padx=(100,0) )
        l_test = tkinter.Label(self.frame3_2, text="select test instance", bg= "gray90")
        combb_test = Combobox(self.frame3_2, values=self.guiFunc.dataset.ClassList, state='readonly', width=10)
        b_predict = tkinter.Button(self.frame3_2, text="predict", command=self.guiFunc.predict, highlightthickness=4,
                      #highlightcolor="SkyBlue4",
                      anchor="center",
                      highlightbackground="gray90",
                      borderwidth=0)
        l_test.grid(row = 0, column = 0, sticky = W )
        combb_test.grid(row = 0, column = 1, sticky = W )
        b_predict.grid(row = 1, column = 0, sticky = W+E+N+S, columnspan = 2)

        '''
        def process_ongoing():
            self.text_features.insert(INSERT, "Extracting data features... \n")
        b3.bind("<Button-1>", process_ongoing)
        '''

    def show_frame(self, StartPageControl, algo):
        if algo == "USEPage":
            frame = USEPage(StartPageControl, self)
        elif algo == "SMAPPage":
            frame = SMAPPage(StartPageControl, self)
        elif algo == "SMAPLBPage":
            frame = SMAPLBPage(StartPageControl, self)
        #frame.grid(row=0, column=0, sticky=W)
        frame.tkraise()


    def plot(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        v = np.array([16, 16.3125, 17.6394, 16.003, 17.2861, 17.3131, 19.1259, 18.9694, 22.0003, 22.81226])
        p = np.array([16.23697, 17.31653, 17.22094, 17.68631, 17.73641, 18.6368,
                      19.32125, 19.31756, 21.20247, 22.41444, 22.11718, 22.12453])

        fig = Figure(figsize=(6, 6))
        a = fig.add_subplot(111)
        a.scatter(v, x, color='red')
        a.plot(p, range(2 + max(x)), color='SkyBlue4')
        a.invert_yaxis()

        a.set_title("Estimation Grid", fontsize=16)
        a.set_ylabel("Y", fontsize=14)
        a.set_xlabel("X", fontsize=14)

        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()

if __name__ == '__main__':
    application = StartPage()
    application.mainloop()

'''
    # <Button-1>: event produced by the left tkinter.Button of mouse
    #frame.bind("<ButtonRelease-1>", popup)
'''


