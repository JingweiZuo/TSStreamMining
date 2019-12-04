''' Navigate to the URL
    http://localhost:5006/draw_TS_Stream
to receive the generated information from the main web application program "ISETS_webapp.py".
'''

from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import Slider, Dropdown
from bokeh.plotting import figure
import os
import utils.utils as util

#Init variables
dataset_folder = '/Users/Jingwei/PycharmProjects/use_reconstruct/SourceCode/ISMAP/ISETS_webapp/uploaded_data/'
t_stamp = 0
window_size = 5
TSClass = None

plot_TS_window = figure(plot_height=150, plot_width=300, title='New Incoming TS micro-batch', tools="reset")
plot_TS_all = figure(plot_height=150, plot_width=300, title='All historical TS')

# Set up widgets, the input file's name should be end with "Train.csv"
filelist = [f for f in os.listdir(dataset_folder) if f.endswith('Train.csv')]
dataset = filelist[0]
ts_list = util.load_dataset_list(dataset_folder+ dataset)
class_repetitiveList = [ts.class_timeseries for ts in ts_list]
classList = list(dict.fromkeys(class_repetitiveList))
menu_C = [(str(t), "Class: "+str(t)) for t in classList]
class_select = Dropdown(label="Select Class", button_type="success", menu=menu_C)
winSize_slider = Slider(start=0, end=20, value=5, step=1, title="Window Size")

def set_class(attr, old, new):
    global TSClass
    TSClass = class_select.value
    class_select.label = class_select.value

def set_windowSize(attr, old, new):
    global window_size
    window_size = winSize_slider.value

class_select.on_change('value', set_class)
winSize_slider.on_change('value', set_windowSize)
# Set up layouts and add to document
widgetSet = widgetbox(class_select, winSize_slider)

def draw_TS():
    global t_stamp, dataset, window_size, plot_TS_window, plot_TS_all, TSClass, dataset_folder

    t_stamp  += window_size
    list_timeseries = util.load_dataset(dataset_folder + dataset)

    name_dataset = {k: v for ds in list_timeseries for k, v in ds.items()}
    dataset_list = list(name_dataset.values())
    # plot_window: input_TSBatch
    # plot_all: TS_set in "ISETS_Web_backend", how to extract the value? refer to a function which returns an object
    plot_TS_window.axis.visible = False
    plot_TS_all.axis.visible = False
    if t_stamp + window_size < 100:
        if plot_TS_window.select({'name': str(t_stamp - window_size)}) != None:
            line1 = plot_TS_window.select({'name': str(t_stamp - window_size)})
            line1.visible = False
        for ts in dataset_list[t_stamp:t_stamp+window_size]:
            if str(int(ts.class_timeseries)) == TSClass.split()[1]:
                x = range(len(ts.timeseries))
                y = ts.timeseries
                plot_TS_window.line(x, y, line_width=1, name = str(t_stamp))
    if t_stamp + window_size < 50:
        if plot_TS_all.select({'name': str(t_stamp - window_size)}) != None:
            line2 = plot_TS_all.select({'name': str(t_stamp - window_size)})
            line2.visible = False
        n = t_stamp + window_size
        for ts in dataset_list[:n]:
            if str(int(ts.class_timeseries)) == TSClass.split()[1]:
                x = range(len(ts.timeseries))
                y = ts.timeseries
                plot_TS_all.line(x, y, line_width=1, name=str(t_stamp))

curdoc().add_root(column(widgetSet, plot_TS_window, plot_TS_all, width=370))
curdoc().add_periodic_callback(draw_TS, 10000)
curdoc().title = "draw_TS_Stream"
