from flask import Flask, flash, redirect, render_template, request, session, abort, jsonify
#from bokeh.embed import autoload_server
from bokeh.embed import server_document, components
from werkzeug.utils import secure_filename
from bokeh.plotting import figure
from bokeh.layouts import gridplot, row, column, widgetbox
from bokeh.models import ColumnDataSource, AjaxDataSource, LinearAxis, Range1d
from bokeh.models.widgets import Slider, TextInput, Button, Dropdown, CheckboxButtonGroup, RadioButtonGroup
from bokeh.io import show, curdoc
from threading import Thread
import pandas as pd
import utils.utils as util
import time
from ISETS_Web_backend import account_api, global_structure

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.register_blueprint(account_api)
thread = None
#data_directory = "/Users/Jingwei/PycharmProjects/distributed_use/SourceCode/TestDataset/UCR_TS_Archive_2015"
#datasetName = data_directory + "/FordA/FordA_TRAIN"


###### Parameters to set for Concept Drift Detection ######
#drift_strategy = "manual_set loss"
#drift_strategy = "mean loss variance"
drift_strategy = "PH test"
###### Parameters for TS stream ######
k = 10
m_ratio = 0.05
stack_ratio = 1
window_size = 5
distance_measure = "mass_v2"

@app.route('/')
def index():
    return 'Hello World!'

@app.route("/dashboard/", methods=['POST', 'GET'])
def hello():
    #script=autoload_server(model=None,app_path="/bokeh-sliders",url="http://localhost:5006")
    bokeh_script=server_document("http://localhost:5006/bokeh-sliders")
    bokeh_server = server_document("http://localhost:5006/bokeh-server")
    pl_conceptDrift = plot_conceptDrift()
    global thread
    if request.method == 'POST':
        file = request.files['file']
        datasetName = secure_filename(file.filename)
        file.save(datasetName)
        # to start a new Thread for computation, how to the transfer the parameters?
        if thread == None:
            thread = Thread(target=global_structure, args=(k, datasetName, m_ratio, stack_ratio, window_size, distance_measure, drift_strategy))
            thread.start()
    return render_template('hello.html',bokS=bokeh_script, bokeh_server=bokeh_server, pl_conceptDrift = pl_conceptDrift)

def plot_conceptDrift():
    # "append" mode is used to output the concatenated data

    source = AjaxDataSource(data_url='http://127.0.0.1:5000/ConceptDrift/', polling_interval=10, mode='append')
    source.data = dict(t_stamp=[], batch_loss=[], avg_loss=[], cum_loss = [],
                       mincum_loss=[], PH=[], drift_num=[],
                       label_batch_loss=[], label_avg_loss=[], label_cum_loss=[], label_mincum_loss=[], label_PH=[],
                       label_concept_drift=[], sys_time=[], memory=[], label_memory=[], nbr_TS=[], label_nbrTS=[])

    if drift_strategy == "manual_set loss":
        plot = figure(plot_height=300, plot_width=500, x_range=(0, 10000), y_range=(0, 1.1))
        plot.line('t_stamp', 'batch_loss', source=source, line_color="blue", legend="label_batch_loss", line_width=2)
        plot.square('t_stamp', 'drift_num', source=source, color="orange", legend="label_concept_drift", size=5)

        plot_mem = figure(plot_height=300, plot_width=500, x_range=(0, 2000), y_range=(0, 100))
        plot_mem.line('sys_time', 'memory', source=source, line_color="blue", legend="label_memory", line_width=2)
        plot_mem.extra_y_ranges = {"foo": Range1d(start=0, end=300)}
        plot_mem.add_layout(LinearAxis(y_range_name="foo"), 'right')
        plot_mem.line('sys_time', 'nbr_TS', source=source, y_range_name="foo", line_color="red", legend="label_nbrTS", line_width=2)

        return components(row(plot, plot_mem))
    elif drift_strategy == "mean loss variance":
        plot = figure(plot_height=300, plot_width=500, x_range=(0, 10000), y_range=(0, 1.1))
        plot.line('t_stamp', 'avg_loss', source=source, line_color="red", legend="label_avg_loss", line_width=2)
        plot.line('t_stamp', 'batch_loss', source=source, line_color="blue", legend="label_batch_loss", line_width=2)
        plot.square('t_stamp', 'drift_num', source=source, color="orange", legend="label_concept_drift", size=5)
        plot_mem = figure(plot_height=300, plot_width=500, x_range=(0, 6000), y_range=(0, 100))
        plot_mem.extra_y_ranges = {"foo": Range1d(start=0, end=300)}
        plot_mem.add_layout(LinearAxis(y_range_name="foo"), 'right')
        plot_mem.line('sys_time', 'memory', source=source, line_color="blue", legend="label_memory", line_width=2)
        plot_mem.line('sys_time', 'nbr_TS', source=source, y_range_name="foo", line_color="red", legend="label_nbrTS", line_width=2)
        return components(row(plot, plot_mem))
    else:
        plot1 = figure(plot_height=300, plot_width=400, x_range=(0, 2200), y_range=(0, 1.1))
        '''plot1.line('t_stamp', 'avg_loss', source=source, line_color="red", legend="label_avg_loss", line_width=2)
        plot1.line('t_stamp', 'batch_loss', source=source, line_color="blue", legend="label_batch_loss", line_width=2)
        plot1.square('t_stamp', 'drift_num', source=source, color="orange", legend="label_concept_drift", size=5)'''
        plot1.line('t_stamp', 'avg_loss', source=source, line_color="red", line_width=2)
        plot1.line('t_stamp', 'batch_loss', source=source, line_color="blue", line_width=2)
        plot1.square('t_stamp', 'drift_num', source=source, color="orange", size=5)
        plot2 = figure(plot_height=300, plot_width=400, x_range=(0, 2200), y_range=(-3, 3))
        #plot2.line('t_stamp', 'cum_loss', source=source, line_color="red", legend="label_cum_loss", line_width=2)
        #plot2.line('t_stamp', 'mincum_loss', source=source, line_color="blue", legend="label_mincum_loss", line_width=2)
        '''plot2.line('t_stamp', 'PH', source=source, line_color="black", legend="label_PH", line_width=2)
        plot2.square('t_stamp', 'drift_num', source=source, color="orange", legend="label_concept_drift", size=5)'''
        plot2.line('t_stamp', 'PH', source=source, line_color="black", line_width=2)
        plot2.square('t_stamp', 'drift_num', source=source, color="orange", size=5)
        plot_mem = figure(plot_height=300, plot_width=400, x_range=(0, 3000), y_range=(0, 100))
        plot_mem.extra_y_ranges = {"foo": Range1d(start=0, end=200)}
        plot_mem.add_layout(LinearAxis(y_range_name="foo"), 'right')
        plot_mem.line('sys_time', 'memory', source=source, line_color="blue", legend="label_memory", line_width=2)
        plot_mem.line('sys_time', 'nbr_TS', source=source, y_range_name="foo", line_color="red", legend="label_nbrTS", line_width=2)
        return components(row(plot1, plot2, plot_mem))

if __name__ == "__main__":
    app.run()
