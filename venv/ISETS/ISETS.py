import numpy as np
import pandas as pd
import utils.similarity_measures as sm
import SMAP.MatrixProfile as mp
import memory_block as mb
import SMAP_block as sb
import evaluation_block as eb
import utils.utils as util
import time, os

drift = False
drift_prev = False

def global_structure(k, data_directory, m_ratio, stack_ratio, window_size, distance_measure, drift_strategy, thresh_loss):

    global drift, drift_prev
    list_timeseries = util.load_dataset(data_directory)
    name_dataset = {k: v for ds in list_timeseries for k, v in ds.items()}
    dataset_list = list(name_dataset.values())

    m = util.min_length_dataset(dataset_list)
    print("Maximum length of shapelet is : " + str(m))
    print("Size of dataset is : " + str(len(dataset_list)))
    min_length = int(0.1 * m)
    max_length = int(0.5 * m)
    m_list = range(min_length, max_length, int(m * m_ratio))
    stack_size = stack_ratio * len(dataset_list)
    TS_set = []
    MP_set_all = {}

    # Initialization of shap_set
    driftDetection = eb.driftDetection()
    '''if window_size==1:
        w = 2
    else:
        w = 1'''

    inputTSBatch = driftDetection.stream_window(dataset_list, window_size)  # Test is OK
    ################# The initial Shapelet Extraction #################
    start = time.time()
    TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list,
                                                    distance_measure)
    end = time.time()
    print("Time cost for extraction on the first batch is : " + str(end - start))
    shap_set = sb.extract_shapelet_all_length(k, TS_set, MP_set_all, m_list)
    print("MARKER1: initial shap_set size is " + str(len(shap_set)))

    #***********# The Output File Configuration #***********#
    drift_col_name = 'LossThresh_' + str(thresh_loss)
    output_driftInfo = pd.DataFrame([[0]], columns=[drift_col_name])
    output_loss = pd.DataFrame([[0,0,0,0,0,0]], columns=['t_stamp', 'loss_batch', 'cum_loss', 'PH', 'avg_loss', 'nbr_drift'])
    output_shapelet = pd.DataFrame([[0,0,0,0,0]], columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])

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
        if drift == True or batch_loss >= thresh_loss:
            ################# Shapelet Update #################
            TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list,
                                                            distance_measure)
            shap_set = sb.extract_shapelet_all_length(k, TS_set, MP_set_all, m_list)

            # ***********# The Output File Configuration #***********#
            for shap in shap_set:
                shap_set = [driftDetection.t_stamp, shap.name, shap.Class, str(shap.subseq), shap.normal_distance]
                shap_pd = pd.DataFrame([shap_set],
                                       columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
                output_shapelet = output_shapelet.append(shap_pd)

            if drift == True:
                drift_prev = True
                print("MARKER3.1: Drift is Ture")
            if batch_loss >= thresh_loss:
                print("MARKER3.2: batch_loss>=thresh_loss")
            print("len(TS_set) is: " + str(len(TS_set)), ", len(MP_set_all) is: " + str(len(MP_set_all)) + "\n")
        elif drift_prev == True:
            # a Concept Transition event, active caching mechanism kick-off
            print("MARKER4: before a Drift transition, len(TS_set) is: " + str(len(TS_set)),
                  ", len(MP_set_all) is: " + str(len(MP_set_all)))
            TS_set, MP_set_all = mb.elastic_caching_mechanism(TS_set, MP_set_all, shap_set, window_size, driftDetection)
            drift_prev = False
            print("After a Drift transition, len(TS_set) is: " + str(len(TS_set)),
                  ", len(MP_set_all) is: " + str(len(MP_set_all)) + "\n")

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

def global_structure_IncrementalTest(dataset_list, m_list, ):
    global drift
    stack_size = stack_ratio * len(dataset_list)
    TS_set = []
    MP_set_all = {}
    drift_prev = None
    drift_curr = None
    #Initialization of shap_set
    driftDetection = eb.driftDetection()
    if window_size==1:
        w = 2
    else:
        w = 1
    inputTSBatch = driftDetection.stream_window(dataset_list, w)  #Test is OK

    TS_newSet, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)
    print("size of MP_set_all is: " + str(len(MP_set_all)))
    shap_set = sb.extract_shapelet_all_length(k, TS_newSet, MP_set_all, m_list)

    #output_driftInfo = pd.DataFrame([[0]], columns=['LossThresh_0.38'])

    #output_loss = pd.DataFrame([[0,0,0,0,0,0]], columns=['t_stamp', 'loss_batch', 'cum_loss', 'PH', 'avg_loss', 'nbr_drift'])
    #output_shapelet = pd.DataFrame([[0,0,0,0,0]], columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
    while driftDetection.t_stamp < len(dataset_list):
        inputTSBatch = driftDetection.stream_window(dataset_list, window_size)
        #drift_curr = driftDetection.ConceptDrift_detection(shap_set, inputTSBatch, drift_strategy)

        #drift, loss_batch, cum_loss, PH, avg_loss = driftDetection.shapelet_matching(shap_set, inputTSBatch)
        #driftInfo = [drift_curr]
        #df_driftInfo = pd.DataFrame([driftInfo], columns=['LossThresh_0.38'])
        #output_driftInfo = output_driftInfo.append(df_driftInfo)
        ############################################################
        ############# Add Concept transition detection #############
        ############################################################
        drift_curr, loss_batch = driftDetection.ConceptDrift_detection(shap_set, inputTSBatch, drift_strategy)
        print("Drift is " + str(drift_curr))
        if drift_curr == True:
            TS_newSet, MP_set_all = mb.memory_cache_all_length(TS_newSet, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)
            shap_set = sb.extract_shapelet_all_length(k, TS_newSet, MP_set_all, m_list)
            '''for shap in shap_set:
                shap_set = [driftDetection.t_stamp, shap.name, shap.Class, str(shap.subseq), shap.normal_distance]
                shap_pd = pd.DataFrame([shap_set], columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
                output_shapelet = output_shapelet.append(shap_pd)'''
            '''drift_prev == True
        elif drift_prev == True:
            # a Concept Transition event, active caching mechanism kick-off
            TS_newSet, MP_set_all= mb.elastic_caching_mechanism(TS_newSet, MP_set_all, shap_set, window_size, driftDetection)
            drift_prev == False

    dataset_folder = '/'.join(data_directory.split('/')[:-1])
    DriftInfofile = dataset_folder + "/DriftInfo.csv"
    files_list = [f for f in os.listdir(dataset_folder) if f.endswith('DriftInfo.csv')]
    if files_list:
        df_old_driftInfo = pd.read_csv(dataset_folder + '/' + files_list[0])
    else:
        df_old_driftInfo = pd.DataFrame([[0]], columns=['Drift'])
    df_old_driftInfo.reset_index(drop=True, inplace=True)
    output_driftInfo.reset_index(drop=True, inplace=True)
    df_old_driftInfo = pd.concat([df_old_driftInfo, output_driftInfo],  axis=1)
    df_old_driftInfo.to_csv(DriftInfofile, index=False)'''

    '''Output Shapelet & Loss in each Time tick'''
    '''output_loss.to_csv(dataset_folder + "/avg_lossMeasure_Drift.csv", index=False)
    output_shapelet.to_csv(dataset_folder + "/avg_lossMeasure_Shapelet.csv", index=False)'''
    print("size of TS_newSet is " + str(len(TS_newSet)))
    return shap_set

if __name__ == "__main__":
    dataset_name = 'ECG5000'
    data_directory = '/Users/Jingwei/PycharmProjects/distributed_use/venv/TestDataset/UCR_TS_Archive_2015/' + dataset_name
    training_data_directory = '/Users/Jingwei/PycharmProjects/distributed_use/venv/TestDataset/UCR_TS_Archive_2015/' + dataset_name + '/' + dataset_name + '_TRAIN'
    testing = '/Users/Jingwei/PycharmProjects/distributed_use/venv/TestDataset/UCR_TS_Archive_2015/' + dataset_name + '/' + dataset_name + '_TEST'
    k = 10
    m_ratio = 0.05
    thresh_loss = 0.5
    stack_ratio = 1
    window_size = 5
    drift_strategy = "PH test"
    distance_measure = "mass_v2"
    global_structure_IncrementalTest(k, training_data_directory, m_ratio, stack_ratio, window_size, distance_measure, drift_strategy, thresh_loss)
