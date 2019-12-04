from flask import Blueprint, jsonify
import pandas as pd
import memory_block as mb
import SMAP_block as sb
import evaluation_block as eb
import utils.utils as util
import psutil as ps
import time

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

cacheData = False
inputTSBatch = []
TS_set = []

def adaptive_feature_extraction(k, train_dataset, m_ratio, stack_ratio, window_size, distance_measure, drift_strategy, thresh_loss):
    dataset_list = util.load_dataset_list(train_dataset)
    ##############################Modified variable for Web GUI##############################
    global t_stamp, batch_loss, avg_loss, cum_loss, mincum_loss, PH, drift, cacheData, inputTSBatch, TS_set, start_time
    drift_prev = False
    dataset_folder = '/Users/Jingwei/PycharmProjects/use_reconstruct/SourceCode/ISMAP/ISETS_webapp/uploaded_data/'
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

    evaluation = eb.evaluation_block()
    inputTSBatch = evaluation.stream_window(dataset_list, window_size)  #Test is OK
    ################# The initial Shapelet Extraction #################
    start = time.time()
    TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)
    end = time.time()

    print("Time cost for extraction on the first batch is : " + str(end-start))
    shap_set = sb.extract_shapelet_all_length(k, TS_set, MP_set_all, m_list)
    print("MARKER1: initial shap_set size is " + str(len(shap_set)))

    # ***********# The Output File Configuration #***********#
    drift_col_name = 'LossThresh_' + str(thresh_loss)
    output_driftInfo = pd.DataFrame([[0, 0]], columns=['t_stamp', drift_col_name])
    output_loss = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0]],
                               columns=['t_stamp', 'drift', 'batch_loss', 'avg_loss', 'cum_loss', 'mincum_loss', 'PH'])
    output_shapelet = pd.DataFrame([[0, 0, 0, 0, 0]],
                                   columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])

    while evaluation.t_stamp < len(dataset_list):
        inputTSBatch = evaluation.stream_window(dataset_list, window_size)
        ################ Detect Concept Drift & Loss ###################
        if drift_strategy == "manual_set loss":
            drift, batch_loss = evaluation.chunk_evaluation(shap_set, inputTSBatch, drift_strategy)
            print("MARKER2: time : "+ str(evaluation.t_stamp) + "drift is : " + str(drift) + ", batch_loss is : " + str(batch_loss))
        elif drift_strategy == "mean loss variance":
            drift, batch_loss, avg_loss = evaluation.chunk_evaluation(shap_set, inputTSBatch, drift_strategy)
            print("MARKER2: time : "+ str(evaluation.t_stamp) + "drift is : " + str(drift) + ", batch_loss is : " + str(batch_loss) + ", avg_loss is : " + str(avg_loss))
        else:
            drift, batch_loss, avg_loss, cum_loss, mincum_loss, PH = evaluation.chunk_evaluation(shap_set, inputTSBatch, drift_strategy)

            # ***********# The Output File Configuration #***********#
            driftInfo = [evaluation.t_stamp, drift]
            df_driftInfo = pd.DataFrame([driftInfo], columns=['t_stamp', drift_col_name])
            output_driftInfo = output_driftInfo.append(df_driftInfo)
            loss_set = [evaluation.t_stamp, drift, batch_loss, avg_loss, cum_loss, mincum_loss, PH]
            loss_pd = pd.DataFrame([loss_set],
                                   columns=['t_stamp', 'drift', 'batch_loss', 'avg_loss', 'cum_loss', 'mincum_loss', 'PH'])
            output_loss = output_loss.append(loss_pd)

            print(
                "MARKER2: time : "+ str(evaluation.t_stamp) + "drift is : " + str(drift) + ", batch_loss is : " + str(batch_loss) + ", avg_loss is : " + str(
                    avg_loss) + ", cum_loss is : " + str(cum_loss) + ", mincum_loss is : " + str(mincum_loss) + ", PH is : " + str(PH))
        t_stamp = evaluation.t_stamp
        # Two cases for updating Shapelet set:
        # 1. Concept Drift detected
        # 2. the observed loss exceeds the threshold, we need to update it under the static concept
        if drift == True or batch_loss>=thresh_loss:
            ################# Shapelet Update #################
            cacheData = True
            TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list,
                                                            distance_measure)
            shap_set = sb.extract_shapelet_all_length(k, TS_set, MP_set_all, m_list)
            if drift == True:
                drift_prev = True
                print("MARKER3.1: Drift is Ture")
            if batch_loss>=thresh_loss:
                print("MARKER3.2: batch_loss>=thresh_loss")
            print("len(TS_set) is: " + str(len(TS_set)), ", len(MP_set_all) is: " + str(len(MP_set_all)) + "\n")
        elif drift_prev == True:
            cacheData = False
            # a Concept Transition event, active caching mechanism kick-off
            print("MARKER4: before a Drift transition, len(TS_set) is: " + str(len(TS_set)),
                  ", len(MP_set_all) is: " + str(len(MP_set_all)))
            TS_set, MP_set_all= mb.elastic_caching_mechanism(TS_set, MP_set_all, shap_set, window_size, evaluation, drift_strategy)
            drift_prev = False
            print("After a Drift transition, len(TS_set) is: " + str(len(TS_set)), ", len(MP_set_all) is: " + str(len(MP_set_all)) + "\n")
        else:
            cacheData = False
        # ***********# The Output File Configuration #***********#
        # 1. Read Shapelet from file
        ShapeletFile = dataset_folder + "/ShapeletFile.csv"
        # 2. Create new shapelet dataframe for new time tick
        for shap in shap_set:
            shap_info = [evaluation.t_stamp, shap.name, shap.Class, str(shap.subseq), shap.normal_distance]
            shap_pd = pd.DataFrame([shap_info],
                                   columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
            output_shapelet = output_shapelet.append(shap_pd)
        # 3. Concatenate old and new shapelet dataframes, and output into file.
        output_shapelet.reset_index(drop=True, inplace=True)
        output_shapelet.to_csv(ShapeletFile, index=False)

    return shap_set

#Concept Drift Data: avg_loss, batch_loss, cum_loss, PH, t_stamp;
#Question: How to receive the information from GUI and react with it?
@account_api.route('/ConceptDrift/', methods=['POST'])
def data_ConceptDrft():
    global t_stamp, batch_loss, avg_loss, cum_loss, mincum_loss, PH, drift, cacheData, inputTSBatch, TS_set, start_time
    if drift == True:
        drift_num = 1
    else:
        drift_num = -10
    if cacheData == True:
        cacheData_num = 1
    else:
        cacheData_num = -10
    mem = ps.virtual_memory().percent
    sys_time = time.time() - start_time
    if start_time == 0:
        mem = 0
        sys_time = 0
    return jsonify(t_stamp=[t_stamp], batch_loss=[batch_loss], avg_loss=[avg_loss], cum_loss=[cum_loss],
                   mincum_loss=[mincum_loss], PH=[PH], drift_num=[drift_num], cacheData_num = [cacheData_num],
                   label_batch_loss=['Lc(t)'], label_avg_loss=['Lavg(t)'], label_cum_loss=['cum_loss'],
                   label_mincum_loss=['mincum_loss'], label_PH=['PH'], label_concept_drift=['concept drift detected'], label_cacheData = ['cache instance'],
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
