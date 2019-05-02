import numpy as np
import similarity_measures as sm
import utils.old_Utils as util
import utils.Shapelet as shap
import scipy

from pyspark import SparkContext, SparkConf
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType, MapType
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, split, hash, col, collect_list, collect_set, _collect_list_doc, _collect_set_doc

# Global variable for the slope of Sigmoid Function
theta = 1
n = 0
loss = 0

master_id = "local"
#master_id = "spark://spark-master:7077"
appname = "distributed_use"
#spark = SparkSession.builder.master(master_id).appName(appname).getOrCreate()
sc = SparkContext(appName=appname)
sc.setLogLevel("INFO")# or "WARN"
#sc.addPyFile("similarity_measures.py")
spark = SparkSession(sc)

#sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x, theta, x0):
    return (1 / (1 + np.exp(-theta*(x-x0))))

def shapelet_matching(shap_set, TS_window):
    # using Sigmoid for Loss Function
    # parallel computing for TSs in the window
    global theta, n, loss
    loss_batch = 0
    w = len(TS_window)
    # Loss Computing, with a fading strategy or not?
    for ts in TS_window:
        shap_set = {s for s in shap_set if s.Class == ts.class_timeseries}
        min_dist = np.inf
        min_s = shap()
        # find the closest Shapelet to 'ts'
        for s in shap_set:
            dist = np.min(sm.calculate_distances(ts, s.subseq, "mass_V2"))
            if min_dist > dist:
                min_dist = dist
                min_s = s
        l = sigmoid(min_dist, theta, min_s.thresh)
        loss_batch += l
        n += 1
    loss = loss * (n-w) / n +  loss_batch/n
    loss_batch = loss_batch / w
    # Check if there's a Concept Drift
    if loss_batch > loss :
        flag = True
    else:
        flag = False
    return 0