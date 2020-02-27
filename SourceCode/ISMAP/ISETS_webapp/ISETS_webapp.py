from flask import Flask, render_template, request
from bokeh.embed import server_document, components
from werkzeug.utils import secure_filename
from bokeh.plotting import figure
from bokeh.layouts import row
from bokeh.models import AjaxDataSource, LinearAxis, Range1d, Span

from threading import Thread
import time, os
from ISETS_webbackend import account_api, adaptive_feature_extraction

import logging
#Desactivate the log output in the terminal
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.register_blueprint(account_api)
thread = None

###### Three strategies for Concept Drift Detection ######
# 1. When newly input batch loss > manually set loss, a Concept Drift is detected
# 2. When newly input batch loss > mean historical loss, a Concept Drift is detected
# 3. Page-Hinkey (PH) test, consider the change point in loss signal as a Concept Drift
#drift_strategy = "manual_set loss"
#drift_strategy = "mean loss variance"
drift_strategy = "PH test"

###### Parameters for TS Stream ######
k = 10
m_ratio = 0.1
stack_ratio = 1
window_size = 5
distance_measure = "mass_v2"
thresh_loss = 0.5
dataset_folder = '/Users/Jingwei/PycharmProjects/use_reconstruct/SourceCode/ISMAP/ISETS_webapp/uploaded_data/'

@app.route('/')
def index():
    return 'Hello World!'

@app.route("/dashboard/", methods=['POST', 'GET'])
def hello():
    import subprocess
    import signal
    process = subprocess.Popen('bokeh serve draw_TS_Stream.py draw_adaptive_shapelets.py --port 5006 --allow-websocket-origin=127.0.0.1:5000',shell=True)
    time.sleep(2)

    draw_TS_Stream = server_document("http://localhost:5006/draw_TS_Stream")
    draw_adaptive_shapelets = server_document("http://localhost:5006/draw_adaptive_shapelets")
    pl_conceptDrift = plot_conceptDrift()
    global thread

    if request.method == 'POST':
        file = request.files['file']
        datasetName = secure_filename(file.filename)
        # clear historical uploaded datasets
        filelist = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]
        for f in filelist:
            os.remove(os.path.join(dataset_folder, f))
        file.save("uploaded_data/" + datasetName)

        process.send_signal(signal.SIGINT)
        os.system('bokeh serve draw_TS_Stream.py draw_adaptive_shapelets.py --port 5006 --allow-websocket-origin=127.0.0.1:5000')

        TrainDataset = dataset_folder + datasetName
        # to start a new Thread for computation, how to the transfer the parameters?
        if thread == None:
            thread = Thread(target=adaptive_feature_extraction, args=(k, TrainDataset, m_ratio, stack_ratio, window_size, distance_measure, drift_strategy, thresh_loss))
            thread.start()
    return render_template('webapp_interface.html', draw_adaptive_shapelets=draw_adaptive_shapelets, draw_TS_Stream=draw_TS_Stream, pl_conceptDrift = pl_conceptDrift)

# Plot the Curves for Concept Drift detection and Caching elimination
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
        plot.square('t_stamp', 'cacheData_num', source=source, color="red", legend="label_cacheData", size=5)

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
        plot.square('t_stamp', 'cacheData_num', source=source, color="red", legend="label_cacheData", size=5)
        plot_mem = figure(plot_height=300, plot_width=500, x_range=(0, 6000), y_range=(0, 100))
        plot_mem.extra_y_ranges = {"foo": Range1d(start=0, end=300)}
        plot_mem.add_layout(LinearAxis(y_range_name="foo"), 'right')
        plot_mem.line('sys_time', 'memory', source=source, line_color="blue", legend="label_memory", line_width=2)
        plot_mem.line('sys_time', 'nbr_TS', source=source, y_range_name="foo", line_color="red", legend="label_nbrTS", line_width=2)
        return components(row(plot, plot_mem))
    else:
        plot1 = figure(plot_height=300, plot_width=350, x_range=(0, 1200), y_range=(0, 1.1))
        plot1.line('t_stamp', 'avg_loss', source=source, line_color="red", legend="label_avg_loss", line_width=2)
        plot1.line('t_stamp', 'batch_loss', source=source, line_color="blue", legend="label_batch_loss", line_width=2)
        plot1.square('t_stamp', 'cacheData_num', source=source, color="red", legend="cacheData", size=1)
        hline1 = Span(location=0.5, dimension='width', line_color='red', line_width=2, line_dash='dashed')
        plot1.add_layout(hline1)

        plot2 = figure(plot_height=300, plot_width=350, x_range=(0, 1200), y_range=(-3, 3))
        #Show the cumunative Loss and minimal cumulative loss, which decide the value of Page-Hinkey (PH) test
        #plot2.line('t_stamp', 'cum_loss', source=source, line_color="red", legend="label_cum_loss", line_width=2)
        #plot2.line('t_stamp', 'mincum_loss', source=source, line_color="blue", legend="label_mincum_loss", line_width=2)
        plot2.line('t_stamp', 'PH', source=source, line_color="black", legend="label_PH", line_width=2)
        plot2.square('t_stamp', 'drift_num', source=source, color="orange", legend="label_concept_drift", size=5)
        hline2 = Span(location=0.4, dimension='width', line_color='black', line_width=2, line_dash='dashed')
        plot2.add_layout(hline2)

        plot_mem = figure(plot_height=300, plot_width=350, x_range=(0, 600), y_range=(0, 100))
        plot_mem.extra_y_ranges = {"foo": Range1d(start=0, end=150)}
        plot_mem.add_layout(LinearAxis(y_range_name="foo"), 'right')
        plot_mem.line('sys_time', 'memory', source=source, line_color="blue", legend="label_memory", line_width=2)
        plot_mem.line('sys_time', 'nbr_TS', source=source, y_range_name="foo", line_color="red", legend="label_nbrTS", line_width=2)

        return components(row(plot1, plot2, plot_mem))

if __name__ == "__main__":
    app.run()
