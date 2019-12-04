import utils.utils as util
import time, os
import numpy as np
import pandas as pd
import utils.similarity_measures as sm
import SMAP.MatrixProfile as mp
import evaluation_block as eb
import memory_block as mb
import SMAP_block as sb

# Adaptive feature extraction considering data source with Concept Drift
def adaptive_feature_extraction(k, data_directory, m_ratio, stack_ratio, window_size, distance_measure, drift_strategy, thresh_loss):
    drift = False
    drift_prev = False
    list_timeseries = util.load_dataset(data_directory)
    name_dataset = {k: v for ds in list_timeseries for k, v in ds.items()}
    dataset_list = list(name_dataset.values())

    m = util.min_length_dataset(dataset_list)
    min_length = int(0.1 * m)
    max_length = int(0.5 * m)
    m_list = range(min_length, max_length, int(m * m_ratio))
    stack_size = stack_ratio * len(dataset_list)
    TS_set = []
    MP_set_all = {}

    # Shapelet set Initialisation
    evaluation = eb.chunk_evaluation()
    inputTSBatch = evaluation.stream_window(dataset_list, window_size)  # Test is OK
    TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)
    shap_set = sb.extract_shapelet_all_length(k, TS_set, MP_set_all, m_list)

    #***********# The Output File Configuration #***********#
    # 1. output_loss: Loss error evolution et Concept Drift detection different time ticks
    # 2. output_shapelet: Generated Shapelets at different time ticks
    # 3. output_caching: The number of TS instances cached in the memory
    output_loss = pd.DataFrame([[0,0,0,0,0,0]], columns=['t_stamp', 'loss_batch', 'cum_loss', 'PH', 'avg_loss', 'nbr_drift'])
    output_shapelet = pd.DataFrame([[0,0,0,0,0]], columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
    output_caching = pd.DataFrame([[0,0]], columns=['t_stamp', 'nbr_TS'])

    while evaluation.t_stamp < len(dataset_list):
        inputTSBatch = evaluation.stream_window(dataset_list, window_size)
        if drift_strategy == "manual_set loss":
            drift, batch_loss = evaluation.chunk_evaluation(shap_set, inputTSBatch, drift_strategy)
            print("MARKER: time : "+ str(evaluation.t_stamp) + "drift is : " + str(drift) + ", batch_loss is : " + str(batch_loss))
        elif drift_strategy == "mean loss variance":
            drift, batch_loss, avg_loss = evaluation.chunk_evaluation(shap_set, inputTSBatch, drift_strategy)
            print("MARKER: time : "+ str(evaluation.t_stamp) + "drift is : " + str(drift) + ", batch_loss is : " + str(batch_loss) + ", avg_loss is : " + str(avg_loss))
        else:
            drift, batch_loss, avg_loss, cum_loss, mincum_loss, PH = evaluation.chunk_evaluation(shap_set, inputTSBatch, drift_strategy)
            # output Loss error and Concept Drift information files
            loss_set = [drift, batch_loss, avg_loss, cum_loss, mincum_loss, PH]
            loss_pd = pd.DataFrame([loss_set],
                                   columns=['drift', 'batch_loss', 'avg_loss', 'cum_loss', 'mincum_loss', 'PH'])
            output_loss = output_loss.append(loss_pd)
            print(
                "MARKER: time : "+ str(evaluation.t_stamp) + "drift is : " + str(drift) + ", batch_loss is : " + str(batch_loss) + ", avg_loss is : " + str(
                    avg_loss) + ", cum_loss is : " + str(cum_loss) + ", mincum_loss is : " + str(mincum_loss) + ", PH is : " + str(PH))

        #Update Shapelet set
        if drift == True or batch_loss >= thresh_loss:
            TS_set, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list,
                                                            distance_measure)
            shap_set = sb.extract_shapelet_all_length(k, TS_set, MP_set_all, m_list)
            # output Shapelet files
            for shap in shap_set:
                shap_set = [evaluation.t_stamp, shap.name, shap.Class, str(shap.subseq), shap.normal_distance]
                shap_pd = pd.DataFrame([shap_set],
                                       columns=['t_stamp', 'shap.name', 'shap.Class', 'shap.subseq', 'shap.score'])
                output_shapelet = output_shapelet.append(shap_pd)
            if drift == True:
                drift_prev = True
        elif drift_prev == True:
            # a Concept Transition event, active caching mechanism kick-off
            TS_set, MP_set_all = mb.elastic_caching_mechanism(TS_set, MP_set_all, shap_set, window_size, evaluation)
            drift_prev = False

        caching_set = [evaluation.t_stamp, len(TS_set)]
        caching_pd = pd.DataFrame([caching_set],
                              columns=['drift', 'batch_loss', 'avg_loss', 'cum_loss', 'mincum_loss', 'PH'])
        output_caching = output_caching.append(caching_pd)

    # ***********# The Output File Configuration #***********#
    dataset_folder = '/'.join(data_directory.split('/')[:-1])
    output_loss.to_csv(dataset_folder + "/lossDriftInfo.csv", index=False)
    output_shapelet.to_csv(dataset_folder + "/adaptiveShapelets.csv", index=False)
    output_caching.to_csv(dataset_folder + "/caching_info.csv", index=False)
    return shap_set