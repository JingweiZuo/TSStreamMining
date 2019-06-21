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
from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, Button, Dropdown, CheckboxButtonGroup, RadioButtonGroup
from bokeh.plotting import figure

import utils.utils as util
# Set up widgets
menu_C = [("Class 1", "Class 1"), ("Class 2", "Class 2"), ("Class 3", "Class 3")]
dropdown1 = Dropdown(label="Select Class", button_type="success", menu=menu_C)

def update_dropdown1(attrname, old, new):
    dropdown1.label = dropdown1.value

dropdown1.on_change('value', update_dropdown1)
button = Button(label="Extract", button_type="success")
# Set up layouts and add to document
input1 = widgetbox(dropdown1, button)

data_directory = "/Users/Jingwei/PycharmProjects/distributed_use/venv/TestDataset/UCR_TS_Archive_2015"
dataset = data_directory + "/FordA/FordA_TRAIN"
t_stamp = 0 
window_size = 10
plot_window = figure(plot_height=150, plot_width=500, title='New Incoming TS micro-batch')
plot_all = figure(plot_height=150, plot_width=500, title='All historical TS')

def draw_TS():
    colors = [
        "#75968f", "red", "blue", "green", "orange",
        "black"
    ]
    global t_stamp, dataset, window_size, plot_window, plot_all
    t_stamp  += 10
    list_timeseries = util.load_dataset(dataset)
    name_dataset = {k: v for ds in list_timeseries for k, v in ds.items()}
    dataset_list = list(name_dataset.values())
    # plot_window: input_TSBatch
    # plot_all: TS_set in "ISETS_Web_backend", how to extract the value? refer to a function which returns an object


    plot_window.axis.visible = False
    plot_all.axis.visible = False
    #historical_TS = forget_degree * len(dataset_list)
    # get the window size
    for ts in dataset_list[t_stamp:t_stamp+window_size]:
        x = range(len(ts.timeseries))
        y = ts.timeseries
        plot_window.line(x, y, line_width=1)
    for ts in dataset_list[:t_stamp+window_size]:
        x = range(len(ts.timeseries))
        y = ts.timeseries
        plot_all.line(x, y, line_width=1)

curdoc().add_root(column(input1, plot_window, plot_all, width=370))
curdoc().add_periodic_callback(draw_TS, 10000)
curdoc().title = "Sliders"
