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
    # parallel computing
    global theta, n
    loss = 0
    loss_batch = 0
    # Loss Computing, with a fading strategy or not?

    for ts in TS_window:
            dist = np.min(sm.calculate_distances(ts, s.subseq, "mass_V2"))
            h = sigmoid(dist, theta, s.thresh)
            if ts.class_timeseries == s.Class:
                y = 1
            else:
                y = 0
            loss_batch += np.abs(h + y -1)
            n += 1
        loss += loss_batch
        loss /= n
        loss_batch /= len(TS_window)

    return 0