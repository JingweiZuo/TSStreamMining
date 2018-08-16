import use.similarity_measures as sm
import sys
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Univariate Shapelet Extraction(USE) evaluation
# distance_measure: brute, mass_v1, mass_v2
def check_performance(list_timeseries, list_shapelets, distance_measure, key='closest|majority'):
    y_pred_maj = []
    y_true = []
    y_pred = []
    true_classification = 0
    false_classification = 0
    for_app = 0
    #instance = 1   # to check the probability for each class prediction
    key = key.split('|')
    # 'list_timeseries': [dict{}, dict{}, ...]
    # 'list_timeseries_dict': dict{ts_name:ts}
    list_timeseries_dict = {k: v for ds in list_timeseries for k, v in ds.items()}
    for timeseries in list_timeseries_dict.values():
        avg_f_dict = defaultdict(int)
        ts_class = timeseries.class_timeseries
        # 'y_true' is the true class of every timeseries in dataset
        y_true.append(ts_class)
        true_classification = false_classification = 0
        min_distance = float('inf')
        min_distance_not_found = float('inf')
        predicted_class_distance = ''
        predicted_class_distance_not_found = ''
        for shap in list_shapelets:
            shap_found, min_dist = pattern_found(timeseries, shap, distance_measure)
            shap_class = shap.class_shapelet
            if(shap_class == timeseries.class_timeseries):
                if shap_found:      #TP
                    #print("shap_found1")
                    true_classification += 1
                    avg_f_dict[shap_class] += 1
                    if(min_dist < min_distance):
                        #print("shap_found1 min_dist < min_distance")
                        min_distance = min_dist
                        predicted_class_distance = shap_class
                else:               #FN
                    #print("shap not found1")
                    false_classification += 1
                    if (min_dist < min_distance_not_found):
                        # print("shap_found1 min_dist < min_distance")
                        min_distance_not_found = min_dist
                        predicted_class_distance_not_found = shap_class

            else:
                if shap_found:      #FP
                    #print("shap_found2")

                    false_classification += 1
                    avg_f_dict[shap_class] += 1
                    if (min_dist < min_distance):
                        #print("shap_found2 min_dist < min_distance")
                        min_distance = min_dist
                        predicted_class_distance = shap_class
                else:               #TN
                    #print("shap not found2")
                    true_classification += 1
                    if (min_dist < min_distance_not_found):
                        # print("shap_found1 min_dist < min_distance")
                        min_distance_not_found = min_dist
                        predicted_class_distance_not_found = shap_class
        #the class is decided by the majority corresponding shapelets in 'shap_list'
        if (key[1] and key[1] == 'majority') or key[0] == 'majority':
            if not predicted_class_distance: #Pattern found
                y_pred_maj.append(predicted_class_distance_not_found)
            else:
                predicted_class = ""
                predicted = 0
                total = 0
                for aKey in avg_f_dict:
                    if avg_f_dict[aKey] > predicted:
                        predicted_class = aKey
                        predicted = avg_f_dict[aKey]
                    total += avg_f_dict[aKey]
                y_pred_maj.append(predicted_class)
            '''
            print("*" * 80)
            print("Time series number", instance, "with name:", timeseries.name)
            print("True Class:", ts_class)
            print("Predicted Classes:")
            for aKey in avg_f_dict:
                print("\t", aKey, ":", round(avg_f_dict[aKey] / total, 2) * 100, "%")
            print("*" * 80)
            instance += 1
            '''
        # the class is decided by the closest shapelet in 'shap_list'
        if key[0] == 'closest':
            if not predicted_class_distance:
                y_pred.append(predicted_class_distance_not_found)
            else:
                y_pred.append(predicted_class_distance)

        if not predicted_class_distance and not predicted_class:
            # can't find any corresponding shapelet in the timeseries, not be able to predict its class
            for_app += 1

    #the proportion of predictable timeseries in the dataset
    app = (len(list_timeseries) - for_app) / float(len(list_timeseries))
    acc=0
    sk_acc = sk_report = sk_acc_maj = sk_report_maj = 0

    if y_pred:
        '''print("y_true is : ")
        print(y_true)
        print("y_pred is : ")
        print(y_pred)'''
        sk_acc = accuracy_score(y_true, y_pred)
        #sk_precision, sk_recall, sk_fscore, sk_support = precision_recall_fscore_support(y_true, y_pred, average='macro')
        sk_report = classification_report(y_true, y_pred)
    if y_pred_maj:
        '''print("y_pred_maj is : ")
        print(y_pred_maj)'''
        sk_acc_maj = accuracy_score(y_true, y_pred_maj)
        #sk_precision_maj, sk_recall_maj, sk_fscore_maj, sk_support_maj = precision_recall_fscore_support(y_true, y_pred_maj, average= 'macro')
    acc = sk_acc
    if sk_acc < sk_acc_maj:
        acc = sk_acc_maj
        sk_report = classification_report(y_true, y_pred_maj)
    # Firstly, need to check if "app=100%", then check other results
    return acc * 100, sk_acc * 100, sk_report, sk_acc_maj * 100, sk_report_maj, app * 100

#distance_measure: brute, mass_v1, mass_v2
#return: shap_found, min_dist
def pattern_found(a_timeseries, a_shapelet, distance_measure):
    pattern_found =False
    min_dist = float('inf')
    if(distance_measure == "mass_v2" ):
        dist_profile = sm.mass_v2(a_timeseries.timeseries, a_shapelet.subsequence)
        min_dist = min(dist_profile)
        #print("min_dist is: " + str(min_dist))
        #print("a_shapelet.dist_threshold: " + str(a_shapelet.dist_threshold))
        if (min_dist <= a_shapelet.dist_threshold):
            pattern_found = True
        else:
            pattern_found = False
    return pattern_found, min_dist


def check_performance_optimized(list_timeseries, list_shapelets, distance_measure, key='closest|majority'):
    y_pred_maj = []
    y_true = []
    y_pred = []
    true_classification = 0
    false_classification = 0
    for_app = 0
    # instance = 1   # to check the probability for each class prediction
    key = key.split('|')
    # 'list_timeseries': [dict{}, dict{}, ...]
    # 'list_timeseries_dict': dict{ts_name:ts}

    list_timeseries_dict = {k: v for ds in list_timeseries for k, v in ds.items()}
    for timeseries in list_timeseries_dict.values():
        avg_f_dict = defaultdict(int)
        ts_class = timeseries.class_timeseries
        # 'y_true' is the true class of every timeseries in dataset
        y_true.append(ts_class)
        true_classification = false_classification = 0
        min_distance = float('inf')
        predicted_class_distance = ''
        shap_list = {}
        for shap in list_shapelets:
            shap_class = shap.class_shapelet
            if shap_class in shap_list.keys():
                shap_list[shap_class].append(shap)
            else:
                shap_list[shap_class] = [shap]
        keys = list(shap_list.copy().keys())
        dist = dict.fromkeys(keys, 0)
        for c, s_list in shap_list.items():
            for s in s_list:
                shap_found, min_dist = pattern_found(timeseries, s, distance_measure)
                dist[c] += min_dist
            dist[c] = dist[c] / len(s_list)
        for s_class, s_dist in dist.items():
            if s_dist == min(dist.values()):
                class_pred = s_class
        y_pred.append(class_pred)
    '''print("y_true is : ")
    print(y_true)
    print("y_pred is : ")
    print(y_pred)'''
    sk_acc = accuracy_score(y_true, y_pred)
    # sk_precision, sk_recall, sk_fscore, sk_support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    sk_report = classification_report(y_true, y_pred)
    return  0, sk_acc * 100, sk_report, 0, 0, 0