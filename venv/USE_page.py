import tkinter
from tkinter import *
from tkinter.ttk import *  # Widgets avec thèmes
from GUI_function import gui_function
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
LARGE_FONT= ("Verdana", 12)

class USE_page(tkinter.Frame):

    def __init__(self, parent, StartPageControl):
        self.ml = StartPageControl.ml
        self.gui = gui_function(self)
        tkinter.Frame.__init__(self, parent)
        #Frame0
        self.frame0 = Frame(self)
        #self.frame1.pack(fill = BOTH, side=TOP)
        self.frame0.pack(fill= tkinter.X)
        self.frame0.config(borderwidth=2, relief=GROOVE)
        #Frame 1
        self.frame1 = Frame(self.frame0)
        self.frame1.grid(row=0,sticky=W)
        l1 = Label(self.frame1, text="Select your testing file: ")
        l2 = Label(self.frame1, text=" ")
        b1 = Button(self.frame1, text="Testing file", command=self.gui.add_testing_file)
        l1.grid(row=0, column=0, sticky=W)
        b1.grid(row=1, column=0, sticky=W)
        l2.grid(row=2, column=0, sticky=W,ipady=10)

        frame2 = Frame(self)
        frame2.pack(fill= tkinter.X)
        b2 = Button(frame2, text="Test", command=self.output_result)
        b2.grid(row=0, column=0, sticky=E)
        b3 = Button(frame2, text="Back to Home", command=lambda: StartPageControl.controller.show_frame(StartPage))
        b3.grid(row=0, column=1, sticky=W)
        frame2.grid_columnconfigure(0, weight=1)
        frame2.grid_columnconfigure(1, weight=1)

    def output_result(self):
        if(self.gui.testing_filename !="file name"):
        #step 1: predict class
            label = tkinter.Label(self, text="Data on processing...", font=LARGE_FONT)
            label.pack(pady=10,padx=10)
            y_predict = self.ml.predict_class(self.gui.testing_filename)
        #step 2: plot the figure
            #step 2.1: plot the original data
            #step 2.2: for different segments in the plot figure, adjust their colors which represent the prediction class
            colors = {
                1: 'blue', #'Transport en camion du Louvre à l’entrepôt Air France', #20 Janv, 10:40-11:25, 21 Janv, 05:25-16:s00
                2: 'green', #'Déchargement du camion',#20 Janv, 11:25-13:30
                3: 'red', #'Transport en zone de fret et chargement de l’avion',#20 Janv,16:48-16:55
                4: 'black', #'Décollage',#20 Janv, 18:31-18:52
                5: 'yellow', #'Vol',#20 Janv, 18:52 - 21 Janv, 01:51
                6: 'fuchsia', #'Atterrissage',#21 Janv, 01:51-02:36
                7: 'sienna', #'Déchagement de l\'avion et chargement du camion' #21 Janv, 03:00-05:25
                8: 'white' #'Déchagement de l\'avion et chargement du camion' #21 Janv, 03:00-05:25
            }

            # length(self.ml.test_df) = 20 * length(y_predict)
            #len_data_i = self.ml.test_df['sum'].size
            #X = np.arange(0, len_data_i, 1)

            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.set_title("Plot figure")
            X = [t for t in self.ml.test_df['time']]
            c = [colors[y] for y in y_predict]
            c = np.repeat(c, 20)
            datemin = self.ml.test_df['time'][0]
            datemax = self.ml.test_df['time'][-1]
            a.set_xlim(datemin, datemax)
            a.scatter(X, self.ml.test_df['sum'], c=c, s=3)

            import matplotlib.patches as mpatches

            classes = ['1.Transportation by truck','2.Unloading the truck','3.Transport in cargo area and loading of the plane', '4.Takeoff', '5.In flight', '6.Landing', '7.Unloading from the plane and loading to the truck', '8.No activity']
            class_colours = ['blue','green','red', 'black', 'yellow', 'fuchsia', 'sienna', 'white']
            recs = []
            for i in range(0,len(class_colours)):
                recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
            a.legend(recs,classes,loc='upper right', fontsize='x-small')
            canvas = FigureCanvasTkAgg(f, self)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
            #a.plot(X, self.ml.test_df['sum'], linewidth=0.5)

            label['text'] = "Processing finished, the prediction result: "