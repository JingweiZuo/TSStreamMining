from SMAP.SMAP import *
import pandas as pd
import matplotlib.pyplot as plt

class timeseries(object):
    def __init__(self):
        self.id = None
        self.Class = ''
        self.seq = None

def drawTS(path, filename):
    tsObjectList1 = []
    tsObjectList2 = []
    testFile = pd.read_csv(path + filename, header=None)
    tsClass1 = testFile[testFile[1] == 1]
    tsClass2 = testFile[testFile[1] == -1]
    for i in tsClass1.index:
        ts = timeseries()
        row = tsClass1.loc[i]
        ts.id = row[0]
        ts.Class = row[1]
        ts.seq = row[2].split(',')
        ts.seq = [float(val) for val in ts.seq]
        tsObjectList1.append(ts)

    for i in tsClass2.index:
        ts = timeseries()
        row = tsClass2.loc[i]
        ts.id = row[0]
        ts.Class = row[1]
        ts.seq = row[2].split(',')
        ts.seq = [float(val) for val in ts.seq]
        tsObjectList2.append(ts)

    return tsObjectList1, tsObjectList2

path = "/Users/Jingwei/Desktop/EDBT demo/USE_results/ECG200/TS_raw/"
desktop = "/Users/Jingwei/Desktop/"
filename = "TS.csv"
tsObjectList1, tsObjectList2 = drawTS(path, filename)

ts = tsObjectList1[0]
seq = ts.seq
plt.title("Source TimeSeries ECG001: " + "normal")
plt.xlabel("index/offset")
plt.ylabel("TS data")
X = range(0, len(seq))
plt.plot(X, seq, color='blue', linewidth=0.5)
plt.savefig(desktop + str(ts.id) + ".eps")
plt.show()

ts = tsObjectList1[1]
seq = ts.seq
plt.title("Target TimeSeries ECG002: " + "normal")
plt.xlabel("index/offset")
plt.ylabel("TS data")
X = range(0, len(seq))
plt.plot(X, seq, color='red', linewidth=0.5)
plt.savefig(desktop + str(ts.id) + ".eps")
plt.show()

DP_all, MP = computeMP(tsObjectList1[0].seq, tsObjectList1[1].seq, 10)

plt.title("Distance Profile at index '0'")
plt.xlabel("offset")
plt.ylabel("Distance")
X = range(0, len(DP_all[0]))
plt.plot(X, DP_all[0], color='red', linewidth=0.5)
plt.savefig(desktop + "DP_demo.eps")
plt.show()

plt.title("Matrix Profile ")
plt.xlabel("offset")
plt.ylabel("Minimum Distance(Nearest Neighor)")
X = range(0, len(MP))
plt.plot(X, MP, color='blue', linewidth=0.5)
plt.savefig(desktop + "MP_demo.eps")
plt.show()