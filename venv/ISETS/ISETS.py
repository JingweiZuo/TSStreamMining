import numpy as np
import pandas as pd
import utils.similarity_measures as sm
import SMAP.MatrixProfile as mp
import memory_block as mb
import SMAP_block as sb
import evaluation_block as eb
import utils.utils as util
import time, sys, os

def global_structure(k, dataset_list, m_list, stack_ratio, window_size, distance_measure, data_directory):
    global drift
    stack_size = stack_ratio * len(dataset_list)
    TS_set = []
    MP_set_all = {}

    #Initialization of shapList
    driftDetection = eb.driftDetection()
    if window_size==1:
        w = 2
    else:
        w = 1
    inputTSBatch = driftDetection.stream_window(dataset_list, w)  #Test is OK
    TS_newSet, MP_newSet_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)

    shapList = sb.extract_shapelet_all_length(k, TS_newSet, MP_newSet_all, m_list)

    output_driftInfo = pd.DataFrame([[0]], columns=['LossThresh_0.38'])

    #output_loss = pd.DataFrame([[0,0,0,0,0,0]], columns=['t_stamp', 'loss_batch', 'cum_loss', 'PH', 'avg_loss', 'nbr_drift'])
    #output_shapelet = pd.DataFrame([[0,0,0,0,0]], columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
    while driftDetection.t_stamp < len(dataset_list):
        inputTSBatch = driftDetection.stream_window(dataset_list, window_size)
        drift = driftDetection.shapelet_matching_incremental(shapList, inputTSBatch, 0.38)
        #drift, loss_batch, cum_loss, PH, avg_loss = driftDetection.shapelet_matching(shapList, inputTSBatch)
        driftInfo = [drift]
        df_driftInfo = pd.DataFrame([driftInfo], columns=['LossThresh_0.38'])
        output_driftInfo = output_driftInfo.append(df_driftInfo)
        '''if drift == True:
            nbr_drift = 1
        else:
            nbr_drift = 0
        loss_set = [driftDetection.t_stamp, loss_batch, cum_loss, PH, avg_loss, nbr_drift]
        loss_pd = pd.DataFrame([loss_set],
                                   columns=['t_stamp', 'loss_batch', 'cum_loss', 'PH', 'avg_loss', 'nbr_drift'])
        output_loss = output_loss.append(loss_pd)'''

        print("Drift is " + str(drift))
        if drift == True:
            TS_newSet, MP_set_all = mb.memory_cache_all_length(TS_newSet, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)
            shapList = sb.extract_shapelet_all_length(k, TS_newSet, MP_set_all, m_list)
            '''for shap in shapList:
                shap_set = [driftDetection.t_stamp, shap.name, shap.Class, str(shap.subseq), shap.normal_distance]
                shap_pd = pd.DataFrame([shap_set], columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
                output_shapelet = output_shapelet.append(shap_pd)'''

    '''Output folder '''
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
    df_old_driftInfo.to_csv(DriftInfofile, index=False)

    '''Output Shapelet & Loss in each Time tick'''
    '''output_loss.to_csv(dataset_folder + "/avg_lossMeasure_Drift.csv", index=False)
    output_shapelet.to_csv(dataset_folder + "/avg_lossMeasure_Shapelet.csv", index=False)'''
    return shapList

'''if __name__ == "__main__":
    k = 10
    data_directory = "/Users/Jingwei/PycharmProjects/distributed_use/venv/TestDataset/UCR_TS_Archive_2015"
    training = "/FordA/FordA_TRAIN"
    testing = "/FordA/FordA_TEST"
    dataset = data_directory + training
    m_ratio = 0.05
    stack_ratio = 1
    window_size = 20
    global_structure(k, dataset, m_ratio, stack_ratio, window_size)'''

