import pandas as pd
from random import *
# For testing the scenario of Time Series Stream, we need to adjust manually the data set to simulate the scenario.
# Sampling the data set "ElectricDevices" into 3 equal-seized subsets with different concepts
df_file_train = pd.read_csv("/Users/Jingwei/PycharmProjects/use_reconstruct/TestDataset/Trace/Trace_TRAIN",header=None)
df_file_test = pd.read_csv("/Users/Jingwei/PycharmProjects/use_reconstruct/TestDataset/Trace/Trace_TEST",header=None)

def df_partition(dataframe):
    class_df = dataframe[0].drop_duplicates(keep='first', inplace=False)
    df_list = []
    class_list = list(class_df)
    class_list.sort()
    for c in class_list:
        df_list.append(dataframe[dataframe[0]==c])
    return df_list, class_list

def concept_construct(df_list, class_list, partition_num):
    n_list = []
    for c in class_list:
        #df_list's index is from 0, c is from 1.
        datasize_c = len(df_list[c-1])
        n_list.append(int(datasize_c/partition_num))
    df_Concept_list = []
    df_Concept_list_origin = []
    for i in range(partition_num):
        df_Concept = pd.DataFrame()
        for c in class_list:
            start_index = i * n_list[c-1]
            end_index = (i+1) * n_list[c-1]
            df_Concept = df_Concept.append(df_list[c-1][start_index:end_index])
            df_Concept.reset_index(drop=True, inplace=True)
        df_Concept_list_origin.append(df_Concept.copy())
        #shift the label for each class distribution/subsets
        df_Concept[0] = df_Concept[0] - i
        index_1 = 0
        index_2 = 0
        for k in range(i):
            index_2 += n_list[k]
        df_Concept[0][index_1:index_2] += len(class_list)
        #Shuffle the dataset to have random class distribution in sequential order
        df_Concept = df_Concept.sample(frac=1).reset_index(drop=True)
        df_Concept_list.append(df_Concept)
    return df_Concept_list_origin, df_Concept_list

def add_noise(df_raw, degree):
    #the noise degree is between 1-10% in normalised TS data
    df_raw[1:] += uniform(-degree, degree)
    return df_raw

def add_noise_random_position(df_raw, degree, period_ratio, aug_time):
    # exept the column of class
    period = period_ratio * (df_raw.shape[1]-1)
    i = 0
    df_full = pd.DataFrame()
    while i < aug_time:
        df_copy = df_raw.copy()
        rand_position = randint(1, df_copy.shape[1])
        end_position = int ( rand_position + period )
        if end_position > df_raw.shape[1] - 1:
            end_position = df_raw.shape[1]
        for j in range(rand_position, end_position):
            df_copy[j] += uniform(-degree, degree)
        i +=1
        df_full = df_full.append(df_copy)
    return df_full

df_noised = add_noise_random_position(df_file_test, 0.01, 0.1, 10)

elemt1, elemt2 = df_partition(df_noised)
df_list_origin, df_list = concept_construct(elemt1, elemt2, 3)

folder_ConceptDriftFile = "/Users/Jingwei/PycharmProjects/use_reconstruct/TestDataset/concept_drift_files/Trace"

DriftFile_full = folder_ConceptDriftFile + "/conceptFull_test.csv"
df_full = pd.DataFrame()
for df in df_list:
    df_full = df_full.append(df)

df_full.to_csv(DriftFile_full, index=False)