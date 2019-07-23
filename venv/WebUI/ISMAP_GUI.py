from flask import Flask, Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, AjaxDataSource
from bokeh.io import show
from testProgram import account_api

app = Flask(__name__)
app.register_blueprint(account_api)

datasetName = ""
class web_plot(object):
    def __init__(self):
        self.loss = {}
        self.cumloss = {}
        self.avgloss = {}
        self.minloss = {}
        self.drift = {}
        self.shapList = {}

@app.route('/')
def index():
    return 'Hello World!'

@app.route('/dashboard/', methods=['POST', 'GET'])
def show_dashboard():
    plots = []
    plots.append(make_plot())
    plots.append(concept_drift_plot())
    global datasetName
    webtitle = "Streaming Time Series Classification by Incremental Shapelet Extraction"
    if request.method == 'POST':
        file = request.files['file']
        datasetName = file.filename
        file.save(datasetName)
    return render_template("dashboard.html", plots= plots, title=webtitle)  # 将拆分好的对象传给模板

def make_plot():
    plot = figure(plot_height=300, plot_width=300, title= 'Drift Detection')

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [2 ** v for v in x]

    plot.line(x, y, line_width=4)

    script, div = components(plot)
    return script, div

def concept_drift_plot():
    # "append" mode is used to output the concatenated data
    source = AjaxDataSource(data_url='http://127.0.0.1:5000/ConceptDrift/', polling_interval=10, mode='append')
    source.data = dict(t_stamp=[], drift_num=[], avg_loss=[], loss_batch=[], label_avg_loss=[], label_loss_batch=[], label_concept_drift=[])
    plot = figure(plot_height=300, plot_width=500,  x_range=(0, 2000), y_range=(0,1.1))
    plot.line('t_stamp', 'avg_loss', source=source, line_color="red", legend="label_avg_loss", line_width=2)
    plot.line('t_stamp', 'loss_batch', source=source, line_color="blue", legend="label_loss_batch", line_width=2)
    plot.square('t_stamp', 'drift_num', source=source, color="orange", legend="label_concept_drift", radius=10)
    script, div = components(plot)
    return script, div

def input_TSbatch_plot() :
    # -> search for another mode to replace the historical data on the plot
    # -> check the format of "source", how to extract the data from "source"?
    source = AjaxDataSource(data_url='http://127.0.0.1:5000/TS_window/',
                            polling_interval=500, mode='append')

    source.data = dict(x=[], y=[])

    plot = figure(plot_height=300, plot_width=300)
    plot.line('x', 'y', source=source, line_width=4)

    script, div = components(plot)
    return script, div

if __name__ == '__main__':
    app.run()
