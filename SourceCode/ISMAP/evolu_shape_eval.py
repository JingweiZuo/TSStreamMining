import matplotlib.pyplot as plt
from utils.utils import *
import pandas as pd

shap_file_dir = '/Users/Jingwei/PycharmProjects/distributed_use/SourceCode/TestDataset/UCR_TS_Archive_2015/Trace'
shap_tol_01_file = shap_file_dir + '/thresh0.5_PH0.5_tole0.1_window5/avg_lossMeasure_Shapelet.csv'
shap_tol_03_file = shap_file_dir + '/thresh0.5_PH0.5_tole0.1_window3/avg_lossMeasure_Shapelet.csv'

shap_tol_01_df = pd.read_csv(shap_tol_01_file)
shap_tol_03_df = pd.read_csv(shap_tol_01_file)

loss_tol_01 = shap_file_dir + '/thresh0.5_PH0.5_tole0.1_window5/avg_lossMeasure_Drift.csv'
loss_tol_03 = shap_file_dir + '/thresh0.5_PH0.5_tole0.3_window5/avg_lossMeasure_Drift.csv'
loss_tol_01_df = pd.read_csv(loss_tol_01)
loss_tol_03_df = pd.read_csv(loss_tol_03)


def select_shapelet(shap_df, t_stamp, Class):
    return shap_df[shap_df["t_stamp"]==t_stamp][shap_df["shap.Class"]==Class]
def draw_shapelet(shap_df, firstK, t_stamp, Class):
    shap_subseq = shap_df['shap.subseq'].tolist()
    shap_score = shap_df['shap.score'].tolist()
    figure = plt.figure(figsize=(8,6), dpi=60)
    i = 1
    for shap in shap_subseq[:firstK]:
        shap_list = shap[1:-1].split()
        shap_list = [float(i) for i in shap_list]
        x = list(range(0, len(shap_list)))
        #ax = figure.add_subplot(int(firstK**0.5),int(firstK**0.5)+1, i)
        ax = figure.add_subplot(111)
        ax.plot(x, shap_list, label="Score: "+ str(round(shap_score[i-1],3)))
        ax.legend()
        i += 1
        #plt.savefig("/Users/Jingwei/Downloads/Shapelet_Time"+str(t_stamp)+"_Class"+str(Class)[:-2]+".eps")
    plt.show()

shap_dict = {340: [], 385: [], 665: [], 795: []}
classList = [1.0, 2.0, 3.0, 4.0]

def convert_file_shapelet(shap_df):
    for t_tick in shap_dict.keys():
        shap_list = []
        for c in classList:
            shap_df_part = select_shapelet(shap_df, classList, c)
            shap_subseq = shap_df_part['shap.subseq'].tolist()
            shap_score = shap_df_part['shap.score'].tolist()
            shap_class = shap_df_part['shap.Class'].tolist()
            for i in range(len(shap_subseq)):
                shap = Shapelet()
                shap.Class = shap_class[i]

                shap_temp = shap_subseq[i][1:-1].split()
                shap.subseq = [float(i) for i in shap_temp]

                shap.dist_threshold = shap_thresh[i]
                shap_list.append(shap)
        shap_dict.update({t_tick: shap_list})
    return shap_dict