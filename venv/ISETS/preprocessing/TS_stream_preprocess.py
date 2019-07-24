import pandas as pd

# For testing the scenario of Time Series Stream, we need to adjust manually the data set to simulate the scenario.
# Sampling the data set "ElectricDevices" into 3 equal-seized subsets with different concepts
df_file_train = pd.read_csv("/Users/Jingwei/PycharmProjects/distributed_use/venv/TestDataset/UCR_TS_Archive_2015/ElectricDevices/ElectricDevices_TRAIN",header=None)
df_file_test = pd.read_csv("/Users/Jingwei/PycharmProjects/distributed_use/venv/TestDataset/UCR_TS_Archive_2015/ElectricDevices/ElectricDevices_TEST",header=None)

def df_partition(dataframe):
    class_list = dataframe[0].drop_duplicates(keep='first', inplace=False)
    df_list = []
    for c in list(class_list):
        df_list.append(dataframe[dataframe[0]==c])
        #print(df)
    return df_list, list(class_list)

def concept_construct(df_list, class_list, partition_num):
    n_list = []
    for i in range(len(class_list)):
        datasize_i = len(df_list[i])
        n_list.append(int(datasize_i/partition_num))
    df_Concept_list = []
    for i in range(partition_num):
        df_Concept = pd.DataFrame()
        for j in range(len(class_list)):
            start_index = i * n_list[j]
            end_index = (i+1) * n_list[j]
            df_Concept = df_Concept.append(df_list[j][start_index:end_index])
            df_Concept.reset_index(drop=True, inplace=True)
        #shift the label for each class distribution/subsets
        df_Concept[0] = df_Concept[0] - i
        index_1 = 0
        index_2 = 0
        for k in range(i):
            index_2 += n_list[k]
        df_Concept[0][index_1:index_2] += len(class_list)
        df_Concept_list.append(df_Concept)
    return df_Concept_list

elemt1, elemt2 = df_partition(df_file_train)
df_list = concept_construct(elemt1, elemt2, 3)
#df_list = [df_Concept1, df_Concept2, df_Concept3]
print(list(df_list[0][0]))