import memory_block as mb
import SMAP_block as sb
import evaluation_block as eb

# Incremental SMAP: take a minimum cost for updating shapelet set, without considering Concept Drift
def ISMAP(k, dataset_list, m_list, stack_ratio, window_size, distance_measure, data_directory):

    stack_size = stack_ratio * len(dataset_list)
    TS_set = []
    MP_set_all = {}
    #Initialization of shap_set
    evaluation = eb.evaluation_block()
    if window_size==1:
        w = 2
    else:
        w = 1
    inputTSBatch = evaluation.stream_window(dataset_list, w)  #Test is OK

    TS_newSet, MP_set_all = mb.memory_cache_all_length(TS_set, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)
    print("size of MP_set_all is: " + str(len(MP_set_all)))
    shap_set = sb.extract_shapelet_all_length(k, TS_newSet, MP_set_all, m_list)

    while evaluation.t_stamp < len(dataset_list):
        inputTSBatch = evaluation.stream_window(dataset_list, window_size)
        import_decision, loss_batch = evaluation.chunk_evaluation(shap_set, inputTSBatch, "manual_set loss")
        print("Import decision is " + str(import_decision))
        if import_decision == True:
            TS_newSet, MP_set_all = mb.memory_cache_all_length(TS_newSet, MP_set_all, stack_size, inputTSBatch, m_list, distance_measure)
            shap_set = sb.extract_shapelet_all_length(k, TS_newSet, MP_set_all, m_list)
    return shap_set

