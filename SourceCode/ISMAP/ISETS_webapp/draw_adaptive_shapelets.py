''' Navigate to the URL
    http://localhost:5006/draw_adaptive_shapelets.py
to receive the generated information from the main web application program "ISETS_webapp.py".
'''

import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models import Legend
from bokeh.models.widgets import Dropdown
from bokeh.plotting import figure
import os
import utils.utils as util

dataset_folder = '/Users/Jingwei/PycharmProjects/use_reconstruct/SourceCode/ISMAP/ISETS_webapp/uploaded_data/'
filelist = [f for f in os.listdir(dataset_folder) if f.endswith('Train.csv')]
dataset = filelist[0]

shap_df = pd.DataFrame([[0, 0, 0, 0, 0]],
                               columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq',
                                        'shap.score'])
# Set up widgets
m = 5 # window size

dataset_list = util.load_dataset_list(dataset_folder + dataset)
class_repetitiveList = [ts.class_timeseries for ts in dataset_list]
classList = list(dict.fromkeys(class_repetitiveList))
t_stampList = range(0, len(dataset_list), m)
menu_T = [(str(t), "TStamp "+str(t)) for t in t_stampList]
menu_C = [(str(t), "Class "+str(t)) for t in classList]

dropdown1 = Dropdown(label="Select Time Stamp", button_type="success", menu=menu_T)
dropdown2 = Dropdown(label="Select Class", button_type="primary", menu=menu_C)
dropdown3 = Dropdown(label="Select Time Stamp", button_type="success", menu=menu_T)
dropdown4 = Dropdown(label="Select Class", button_type="primary", menu=menu_C)
dropdown5 = Dropdown(label="Select Time Stamp", button_type="success", menu=menu_T)
dropdown6 = Dropdown(label="Select Class", button_type="primary", menu=menu_C)
class_1 = '0'
class_2 = '0'
class_3 = '0'
t_stamp1 = '0'
t_stamp2 = '0'
t_stamp3 = '0'

plot1 = figure(plot_height=200, plot_width=300, title='Shapelet (Feature) Ranking', x_range=(0, 100), y_range=(0, 30))
plot2 =figure(plot_height=200, plot_width=300, title='Shapelet (Feature) Ranking', x_range=(0, 100), y_range=(0, 30))
plot3 = figure(plot_height=200, plot_width=300, title='Shapelet (Feature) Ranking', x_range=(0, 100), y_range=(0, 30))
plot1.yaxis.visible = False
plot2.yaxis.visible = False
plot3.yaxis.visible = False
legend1 = Legend(items=[], location='top_right', spacing=-3, label_text_font_size='6pt')
legend2 = Legend(items=[], location='top_right', spacing=-3, label_text_font_size='6pt')
legend3 = Legend(items=[], location='top_right', spacing=-3, label_text_font_size='6pt')
plot1.add_layout(legend1, 'right')
plot2.add_layout(legend2, 'right')
plot3.add_layout(legend3, 'right')

plot1List = {}
plot2List = {}
plot3List = {}

def select_shapelet(t_stamp, Class):
    global shap_df, t_stampList
    files_list = [f for f in os.listdir(dataset_folder) if f.endswith('ShapeletFile.csv')]
    if files_list:
        shap_df = pd.read_csv(dataset_folder + '/' + files_list[0])
    else:
        shap_df = pd.DataFrame([[0, 0, 0, 0, 0]],
                               columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq',
                                        'shap.score'])

    return shap_df[(shap_df["t_stamp"]==int(t_stamp)) & (shap_df["shap.Class"]==float(Class))]

def draw_shapelet(shap_df, firstK, t_stamp, Class, plot, nbr_plot):
    shap_subseq = shap_df['shap.subseq'].tolist()
    shap_score = shap_df['shap.score'].tolist()

    colors = [
        "#75968f", "red", "blue", "green", "orange",
        "black"
    ]
    i = 1
    r = []
    global plot1, plot2, plot3

    for shap in shap_subseq[:firstK]:
        shap_list = shap[1:-1].split()
        shap_list = [float(val)+ 30-5*i for val in shap_list]
        x = list(range(0, len(shap_list)))
        if nbr_plot == 1:
            oldShapelet = plot.select({'name': 'plot1' + str(i-1)})
            oldShapelet.visible = False
            r.append(plot.line(x, shap_list, line_width=2, line_color=colors[i], name='plot1' + str(i-1)))
        elif nbr_plot == 2:
            oldShapelet = plot.select({'name': 'plot2' + str(i-1)})
            oldShapelet.visible = False
            r.append(plot.line(x, shap_list, line_width=2, line_color=colors[i], name='plot2' + str(i - 1)))
        elif nbr_plot == 3:
            oldShapelet = plot.select({'name': 'plot3' + str(i-1)})
            oldShapelet.visible = False
            r.append(plot.line(x, shap_list, line_width=2, line_color=colors[i], name='plot3' + str(i - 1)))
        plot.legend.spacing = -3
        i += 1
    plot.legend.items = [
        (str(round(shap_score[0],3)), [r[0]]),
        (str(round(shap_score[1],3)), [r[1]]),
        (str(round(shap_score[2],3)), [r[2]]),
        (str(round(shap_score[3],3)), [r[3]]),
        (str(round(shap_score[4],3)), [r[4]]),
    ]

def update_dropdown1(attrname, old, new):
    global t_stamp1
    dropdown1.label = dropdown1.value
    t_stamp1 = dropdown1.value.split()[1]
def update_dropdown2(attrname, old, new):
    global class_1, t_stamp1, plot1
    dropdown2.label = dropdown2.value
    class_1 = dropdown2.value.split()[1]
    draw_shapelet(select_shapelet(t_stamp1, class_1), 5, t_stamp1, class_1, plot1, 1)
def update_dropdown3(attrname, old, new):
    global t_stamp2
    dropdown3.label = dropdown3.value
    t_stamp2 = dropdown3.value.split()[1]
def update_dropdown4(attrname, old, new):
    global class_2, t_stamp2, plot2
    dropdown4.label = dropdown4.value
    class_2 = dropdown4.value.split()[1]
    draw_shapelet(select_shapelet(t_stamp2, class_2), 5, t_stamp2, class_2, plot2, 2)
def update_dropdown5(attrname, old, new):
    global t_stamp3
    dropdown5.label = dropdown5.value
    t_stamp3 = dropdown5.value.split()[1]
def update_dropdown6(attrname, old, new):
    global class_3, t_stamp3, plot3
    dropdown6.label = dropdown6.value
    class_3 = dropdown6.value.split()[1]
    draw_shapelet(select_shapelet(t_stamp3, class_3), 5, t_stamp3, class_3, plot3, 3)

dropdown1.on_change('value', update_dropdown1)
dropdown2.on_change('value', update_dropdown2)
dropdown3.on_change('value', update_dropdown3)
dropdown4.on_change('value', update_dropdown4)
dropdown5.on_change('value', update_dropdown5)
dropdown6.on_change('value', update_dropdown6)
# Set up layouts and add to document
input1 = widgetbox(dropdown1, dropdown2)
input2 = widgetbox(dropdown3, dropdown4)
input3 = widgetbox(dropdown5, dropdown6)

curdoc().add_root(row(column(input1, plot1, width=400), column(input2, plot2, width=400), column(input3, plot3, width=400)))
curdoc().title = "draw_adaptive_shapelets"

