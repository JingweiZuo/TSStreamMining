''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np
import pandas as pd
from bokeh.io import curdoc, reset_output
from bokeh.layouts import row, column, widgetbox
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import Slider, Dropdown, CheckboxButtonGroup, RadioButtonGroup
from bokeh.plotting import figure
import os
import utils.utils as util
#Init variables

dataset_folder = '/Users/Jingwei/PycharmProjects/use_reconstruct/SourceCode/ISMAP/ISETS_webapp/uploaded_data/'

#dataset_folder = "/Users/Jingwei/PycharmProjects/use_reconstruct/TestDataset/Trace"
#dataset = dataset_folder + '/' + dataset_folder.split('/')[-1] + '_TRAIN'
t_stamp = 0
window_size = 5
forget_degree = 0
TSClass = None
plot_window = figure(plot_height=150, plot_width=300, title='New Incoming TS micro-batch', tools="reset")
plot_all = figure(plot_height=150, plot_width=300, title='All historical TS')

# Set up widgets
filelist = [f for f in os.listdir(dataset_folder)]
dataset = filelist[0]
ts_list = util.load_dataset_list(dataset_folder+ dataset)
class_repetitiveList = [ts.class_timeseries for ts in ts_list]
classList = list(dict.fromkeys(class_repetitiveList))
menu_C = [(str(t), "Class "+str(t)) for t in classList]

class_select = Dropdown(label="Select Class", button_type="success", menu=menu_C)
winSize_slider = Slider(start=0, end=20, value=5, step=1, title="Window Size")
fDegree_slider = Slider(start=0, end=20, value=5, step=1, title="Forgetting Degree")

def set_class(attr, old, new):
    global TSClass
    TSClass = class_select.value
    class_select.label = class_select.value
def set_windowSize(attr, old, new):
    global window_size
    window_size = winSize_slider.value
def set_forgetDegree(attr, old, new):
    global forget_degree
    forget_degree = fDegree_slider.value
class_select.on_change('value', set_class)
winSize_slider.on_change('value', set_windowSize)
fDegree_slider.on_change('value', set_forgetDegree)
# Set up layouts and add to document
widgetSet = widgetbox(class_select, winSize_slider, fDegree_slider)

def draw_TS():
    global t_stamp, dataset, window_size, plot_window, plot_all, TSClass


    t_stamp  += window_size
    list_timeseries = util.load_dataset(dataset)
    name_dataset = {k: v for ds in list_timeseries for k, v in ds.items()}
    dataset_list = list(name_dataset.values())
    # plot_window: input_TSBatch
    # plot_all: TS_set in "ISETS_Web_backend", how to extract the value? refer to a function which returns an object
    plot_window.axis.visible = False
    plot_all.axis.visible = False
    if t_stamp + window_size < 100:
        if plot_window.select({'name': str(t_stamp-window_size)}) != None:
            line1 = plot_window.select({'name': str(t_stamp-window_size)})
            line1.visible = False
        for ts in dataset_list[t_stamp:t_stamp+window_size]:
            if str(int(ts.class_timeseries)) == TSClass.split(': ')[1]:
                x = range(len(ts.timeseries))
                y = ts.timeseries
                plot_window.line(x, y, line_width=1, name = str(t_stamp))

    if t_stamp + window_size < 50:
        if plot_all.select({'name': str(t_stamp-window_size)}) != None:
            line2 = plot_all.select({'name': str(t_stamp-window_size)})
            line2.visible = False
        n = t_stamp + window_size
        for ts in dataset_list[:n]:
            if str(int(ts.class_timeseries)) == TSClass.split(': ')[1]:
                x = range(len(ts.timeseries))
                y = ts.timeseries
                plot_all.line(x, y, line_width=1, name=str(t_stamp))

curdoc().add_root(column(widgetSet, plot_window, plot_all, width=370))
curdoc().add_periodic_callback(draw_TS, 10000)
curdoc().title = "Sliders"
