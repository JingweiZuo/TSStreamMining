from flask import Flask, Blueprint, render_template, request, jsonify
from threading import Thread
import numpy as np
import pandas as pd
import SMAP.MatrixProfile as mp
import memory_block as mb
import SMAP_block as sb
import evaluation_block as eb
import utils.utils as util
import psutil as ps
import time, sys, os

account_api = Blueprint('account_api', __name__)
# Global variable for output GUI
thread = None
t_stamp = 0
start_time = 0

batch_loss = 0.0
avg_loss = 0.0
cum_loss = 0.0
mincum_loss = 0.0 
PH = 0.0

drift = False
drift_prev = False
inputTSBatch = []
TS_set = []

#Parameter to configure
thresh_loss = 0.5

'''def global_structure(k, data_directory, m_ratio, stack_ratio, window_size, distance_measure):
    list_timeseries = util.load_dataset(data_directory)
    name_dataset = {k: v for ds in list_timeseries for k, v in ds.items()}
    dataset_list = list(name_dataset.values())
    ##############################Modified variable for Web GUI##############################
    global drift, batch_loss, avg_loss, t_stamp, inputTSBatch, TS_set
    min_m = util.min_length_dataset(dataset_list)
    print("Maximum length of shapelet is : " + str(min_m))
    min_length = int(0.1 * min_m)
    max_length = int(0.5 * min_m)
    m_list =range(min_length, max_length, int(min_m * m_ratio))
    stack_size = int(stack_ratio * len(dataset_list))
    TS_set = []
    MP_set_all = {}

    #Initialization of shapList
    driftDetection = eb.driftDetection()
    inputTSBatch = driftDetection.stream_window(dataset_list, window_size)
    TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)

    shapList = sb.extract_shapelet_all_length(k, TS_set, MP_set_all, m_list)
    #output_loss = pd.DataFrame([[0,0,0,0,0,0]], columns=['t_stamp', 'batch_loss', 'cum_loss', 'PH', 'avg_loss', 'nbr_drift'])
    #output_shapelet = pd.DataFrame([[0,0,0,0,0]], columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
    while driftDetection.t_stamp < len(dataset_list):
        inputTSBatch = driftDetection.stream_window(dataset_list, window_size)
        drift, batch_loss, cum_loss, PH, avg_loss = driftDetection.PHtest_detection(shapList, inputTSBatch)
        t_stamp = driftDetection.t_stamp
        if drift == True:
            nbr_drift = 1
        else:
            nbr_drift = 0
            time.sleep(1)
        loss_set = [driftDetection.t_stamp, batch_loss, cum_loss, PH, avg_loss, nbr_drift]
        'loss_pd = pd.DataFrame([loss_set],
                                   columns=['t_stamp', 'batch_loss', 'cum_loss', 'PH', 'avg_loss', 'nbr_drift'])
        output_loss = output_loss.append(loss_pd)'
        if drift == True:
            #TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)
            TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)
            shapList = sb.extract_shapelet_all_length(k, TS_set, MP_set_all, m_list)
            for shap in shapList:
                shap_set = [driftDetection.t_stamp, shap.name, shap.Class, str(shap.subseq), shap.normal_distance]
                shap_pd = pd.DataFrame([shap_set], columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
                output_shapelet = output_shapelet.append(shap_pd)'''


def global_structure(k, data_directory, m_ratio, stack_ratio, window_size, distance_measure, drift_strategy):
    list_timeseries = util.load_dataset(data_directory)
    name_dataset = {k: v for ds in list_timeseries for k, v in ds.items()}
    dataset_list = list(name_dataset.values())

    ##############################Modified variable for Web GUI##############################
    global t_stamp, batch_loss, avg_loss, cum_loss, mincum_loss, PH, drift, drift_prev, inputTSBatch, TS_set, start_time
    start_time = time.time()
    m = util.min_length_dataset(dataset_list)
    print("Maximum length of shapelet is : " + str(m))
    print("Size of dataset is : " + str(len(dataset_list)))
    min_length = int(0.1 * m)
    max_length = int(0.5 * m)
    m_list =range(min_length, max_length, int(m * m_ratio))
    stack_size = stack_ratio * len(dataset_list)
    TS_set = []
    MP_set_all = {}

    #Initialization of shap_set
    driftDetection = eb.driftDetection()
    '''if window_size==1:
        w = 2
    else:
        w = 1'''
    inputTSBatch = driftDetection.stream_window(dataset_list, window_size)  #Test is OK
    ################# The initial Shapelet Extraction #################
    start = time.time()
    TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)
    end = time.time()
    print("Time cost for extraction on the first batch is : " + str(end-start))
    shap_set = sb.extract_shapelet_all_length(k, TS_set, MP_set_all, m_list)
    print("MARKER1: initial shap_set size is " + str(len(shap_set)))

    # ***********# The Output File Configuration #***********#
    drift_col_name = 'LossThresh_' + str(thresh_loss)
    output_driftInfo = pd.DataFrame([[0]], columns=[drift_col_name])
    output_loss = pd.DataFrame([[0, 0, 0, 0, 0, 0]],
                               columns=['t_stamp', 'loss_batch', 'cum_loss', 'PH', 'avg_loss', 'nbr_drift'])
    output_shapelet = pd.DataFrame([[0, 0, 0, 0, 0]],
                                   columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])

    while driftDetection.t_stamp < len(dataset_list):
        inputTSBatch = driftDetection.stream_window(dataset_list, window_size)
        ################ Detect Concept Drift & Loss ###################
        if drift_strategy == "manual_set loss":
            #start = time.time()
            drift, batch_loss = driftDetection.ConceptDrift_detection(shap_set, inputTSBatch, drift_strategy)
            #end = time.time()
            #print("Time cost for extraction on t_stamp "+str(driftDetection.t_stamp) + " is : " + str(end - start))
            print("MARKER2: time : "+ str(driftDetection.t_stamp) + "drift is : " + str(drift) + ", batch_loss is : " + str(batch_loss))
        elif drift_strategy == "mean loss variance":
            drift, batch_loss, avg_loss = driftDetection.ConceptDrift_detection(shap_set, inputTSBatch, drift_strategy)
            print("MARKER2: time : "+ str(driftDetection.t_stamp) + "drift is : " + str(drift) + ", batch_loss is : " + str(batch_loss) + ", avg_loss is : " + str(avg_loss))
        else:
            drift, batch_loss, avg_loss, cum_loss, mincum_loss, PH = driftDetection.ConceptDrift_detection(shap_set, inputTSBatch, drift_strategy)

            # ***********# The Output File Configuration #***********#
            driftInfo = [drift]
            df_driftInfo = pd.DataFrame([driftInfo], columns=[drift_col_name])
            output_driftInfo = output_driftInfo.append(df_driftInfo)
            loss_set = [drift, batch_loss, avg_loss, cum_loss, mincum_loss, PH]
            loss_pd = pd.DataFrame([loss_set],
                                   columns=['drift', 'batch_loss', 'avg_loss', 'cum_loss', 'mincum_loss', 'PH'])
            output_loss = output_loss.append(loss_pd)

            print(
                "MARKER2: time : "+ str(driftDetection.t_stamp) + "drift is : " + str(drift) + ", batch_loss is : " + str(batch_loss) + ", avg_loss is : " + str(
                    avg_loss) + ", cum_loss is : " + str(cum_loss) + ", mincum_loss is : " + str(mincum_loss) + ", PH is : " + str(PH))
        t_stamp = driftDetection.t_stamp
        ############# Add Concept transition detection #############
        # 1. Output the memory cost: the number of instances cached, new time stamp for historical cached data -> another plot
            # 1.1 when eliminating the caching, t_stamp will increase, but there will be no new input instance,
            # 1.2 output the program's memory cost, as well as the number of cached instances, BETTER!
        # 2. Output the eliminating process: markers of elimination
        # 3. In memory_block, replace the fixed stack_size by the elastic caching mechanism -> just set "stack_sie = 1"

        if drift == True or batch_loss>=thresh_loss:
            ################# Shapelet Update #################
            start = time.time()
            TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list,
                                                            distance_measure)
            shap_set = sb.extract_shapelet_all_length(k, TS_set, MP_set_all, m_list)
            end = time.time()
            #print("Time cost for extraction on t_stamp "+str(driftDetection.t_stamp) + " is : " + str(end - start))

            # ***********# The Output File Configuration #***********#
            for shap in shap_set:
                shap_set = [driftDetection.t_stamp, shap.name, shap.Class, str(shap.subseq), shap.normal_distance]
                shap_pd = pd.DataFrame([shap_set],
                                       columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
                output_shapelet = output_shapelet.append(shap_pd)

            if drift == True:
                drift_prev = True
                print("MARKER3.1: Drift is Ture")
            if batch_loss>=thresh_loss:
                print("MARKER3.2: batch_loss>=thresh_loss")
            print("len(TS_set) is: " + str(len(TS_set)), ", len(MP_set_all) is: " + str(len(MP_set_all)) + "\n")
        elif drift_prev == True:
            # a Concept Transition event, active caching mechanism kick-off
            print("MARKER4: before a Drift transition, len(TS_set) is: " + str(len(TS_set)),
                  ", len(MP_set_all) is: " + str(len(MP_set_all)))
            TS_set, MP_set_all= mb.elastic_caching_mechanism(TS_set, MP_set_all, shap_set, window_size, driftDetection)
            drift_prev = False
            print("After a Drift transition, len(TS_set) is: " + str(len(TS_set)), ", len(MP_set_all) is: " + str(len(MP_set_all)) + "\n")

    # ***********# The Output File Configuration #***********#
    dataset_folder = '/'.join(data_directory.split('/')[:-1])
    DriftInfofile = dataset_folder + "/DriftInfo.csv"
    files_list = [f for f in os.listdir(dataset_folder) if f.endswith('DriftInfo.csv')]
    if files_list:
        df_old_driftInfo = pd.read_csv(dataset_folder + '/' + files_list[0])
    else:
        df_old_driftInfo = pd.DataFrame([[0]], columns=['Drift'])
    df_old_driftInfo.reset_index(drop=True, inplace=True)
    output_driftInfo.reset_index(drop=True, inplace=True)
    df_old_driftInfo = pd.concat([df_old_driftInfo, output_driftInfo], axis=1)
    df_old_driftInfo.to_csv(DriftInfofile, index=False)
    # ***********# The Output File Configuration #***********#
    # '''Output Shapelet & Loss in each Time tick'''
    output_loss.to_csv(dataset_folder + "/avg_lossMeasure_Drift.csv", index=False)
    output_shapelet.to_csv(dataset_folder + "/avg_lossMeasure_Shapelet.csv", index=False)
    return shap_set

#Concept Drift Data: avg_loss, batch_loss, cum_loss, PH, t_stamp;
#Question: How to receive the information from GUI and react with it?
@account_api.route('/ConceptDrift/', methods=['POST'])
def data_ConceptDrft():
    global t_stamp, batch_loss, avg_loss, cum_loss, mincum_loss, PH, drift, inputTSBatch, TS_set, start_time
    if drift == True:
        drift_num = 1
    else:
        drift_num = -1
    mem = ps.virtual_memory().percent
    sys_time = time.time() - start_time
    if start_time == 0:
        mem = 0
        sys_time = 0
    return jsonify(t_stamp=[t_stamp], batch_loss=[batch_loss], avg_loss=[avg_loss], cum_loss=[cum_loss],
                   mincum_loss=[mincum_loss], PH=[PH], drift_num=[drift_num],
                   label_batch_loss=['batch_loss'], label_avg_loss=['avg_loss'], label_cum_loss=['cum_loss'],
                   label_mincum_loss=['mincum_loss'], label_PH=['PH'], label_concept_drift=['concept drift area'],
                   sys_time=[sys_time], memory = [mem], label_memory=['memory cost'],
                   nbr_TS=[len(TS_set)], label_nbrTS=['nbr. TS cached'])

#TS data in new Window
#Question: How to read the window size and change the TS data shown in the GUI?
@account_api.route('/TSWindow/', methods=['POST'])
def data_TSWindow():
    global inputTSBatch, TS_set
    # Convert TS object to a serializable object (i.e. String)
    list_inputTS = []
    list_TSset = []

    for inputTS in inputTSBatch:
        list_inputTS.append(str(inputTS.timeseries))
    for TS in TS_set:
        list_TSset.append(str(TS.timeseries))
    return jsonify(inputTSBatch=[';'.join(list_inputTS)], TS_set=[';'.join(list_TSset)])
