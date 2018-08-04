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
    #instance = 1   # to check the probability for each class prediction
    key = key.split('|')

    for timeseries in list_timeseries:
        avg_f_dict = defaultdict(int)
        ts_class = timeseries.class_timeseries
        # 'y_true' is the true class of every timeseries in dataset
        y_true.append(ts_class)
        true_classification = false_classification = 0
        min_distance = float('inf')
        predicted_class_distance = ''
        for shap in list_shapelets:
            shap_found, min_dist = pattern_found(timeseries, shap, distance_measure)
            shap_class = shap.class_shapelet
            if(shap_class == timeseries.class_timeseries):
                if shap_found:      #TP
                    true_classification += 1
                    avg_f_dict[shap_class] += 1
                    if(min_dist < min_distance):
                        min_distance = min_dist
                        predicted_class_distance = shap_class
                else:               #FN
                    false_classification += 1
            else:
                if shap_found:      #FP
                    false_classification += 1
                    avg_f_dict[sequence_class] += 1
                    if (min_dist < min_distance):
                        min_distance = min_dist
                        predicted_class_distance = shap_class
                else:               #TN
                    true_classification += 1
        #the class is decided by the majority corresponding shapelets in 'shap_list'
        if (key[1] and key[1] == 'majority') or key[0] == 'majority':
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
            y_pred.append(predicted_class_distance)

        if not predicted_class_distance and not predicted_class:
            # can't find any corresponding shapelet in the timeseries, not be able to predict its class
            for_app += 1
    #the proportion of predictable timeseries in the dataset
    app = (len(list_timeseries) - for_app) / float(len(list_timeseries))

    sk_acc = sk_report = sk_acc_maj = sk_report_maj = 0
    if y_pred:
        sk_acc = accuracy_score(y_true, y_pred)
        sk_precision, sk_recall, sk_fscore, sk_support = precision_recall_fscore_support(y_true, y_pred,
                                                                                         average='macro')
        sk_report = classification_report(y_true, y_pred)
    if y_pred_maj:
        sk_acc_maj = accuracy_score(y_true, y_pred_maj)
        sk_precision_maj, sk_recall_maj, sk_fscore_maj, sk_support_maj = precision_recall_fscore_support(y_true, y_pred_maj, average= 'macro')
    acc = sk_acc
    if sk_acc < sk_acc_maj:
        acc = sk_acc_maj
        sk_report = classification_report(y_true, y_pred_maj)
    # Firstly, need to check if "app=100%", then check other results
    return acc * 100, sk_acc * 100, sk_report, sk_acc_maj * 100, sk_report_maj, app * 100

#distance_measure: brute, mass_v1, mass_v2
#return: shap_found, min_dist
def pattern_found(a_timeseries, a_shapelet, distance_measure):
    if(distance_measure == "mass_v2" ):
        dist_profile = sm.mass_v2(a_timeseries, a_shapelet)
        min_dist = min(dist_profile)
        if (min_dist <= a_shapelet.dist_threshold):
            pattern_found = True
        else:
            pattern_found = False
    return pattern_found, min_dist
