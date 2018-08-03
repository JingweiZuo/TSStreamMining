import venv.use.similarity_measures as sm
import sys
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Univariate Shapelet Extraction(USE) evaluation
# distance_measure: brute, mass_v1, mass_v2
def check_performance(list_timeseries, list_shapelets, distance_measure, key='closest|majority'):
    for timeseries in list_timeseries:
        for shap in list_shapelets:
            shap_found = pattern_found(timeseries, shap, distance_measure)
            shap_class = shap.class_shapelet
            if(shap_class == timeseries.class_timeseries):
                if shap_found:      #TP
                    true_classification += 1
                else:
                    #FN
                    false_classification += 1
            else:
                if shap_found:      #FP
                    false_classification += 1
                else:
                    #TN
                    true_classification += 1

    tc = fc = 0
    avg_length = 0
    y_pred_maj = []
    y_true = []
    y_pred = []
    instance = 1
    for_app = 0
    key = key.split('|')
    for timeseries in list_timeseries:
        avg_f_dict = defaultdict(int)

        ts_class = timeseries.class_timeseries
        min_seq_length = sys.maxsize
        true_classification = false_classification = 0
        predicted_class_distance = ''
        min_distance = sys.maxsize
        y_true.append(ts_class)

        tp_bool = False
        for a_sequence in list_sequences:
            sequence_class = a_sequence.sequence[0].class_shapelet
            seq_found, win_length, min_dist = pattern_found(timeseries, a_sequence,
                                                            distance_measure=distance_measure)
            if sequence_class == ts_class:
                if seq_found:  # TP
                    # For Accuracy
                    true_classification += 1

                    avg_f_dict[sequence_class] += 1

                    # For Earliness
                    min_seq_length = min(min_seq_length, win_length + a_sequence.length_of_shapelets())

                    tp_bool = True

                    if min_dist <= min_distance:
                        min_distance = min_dist
                        predicted_class_distance = sequence_class

                else:  # FN
                    # For Accuracy
                    false_classification += 1

            else:
                if seq_found:  # FP
                    # For Accuracy
                    false_classification += 1

                    avg_f_dict[sequence_class] += 1

                    if min_dist < min_distance:
                        min_distance = min_dist
                        predicted_class_distance = sequence_class

                else:  # TN
                    # For Accuracy
                    true_classification += 1

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
            print("*" * 80)
            print("Time series number", instance, "with name:", timeseries.name)
            print("True Class:", ts_class)
            print("Predicted Classes:")
            for aKey in avg_f_dict:
                print("\t", aKey, ":", round(avg_f_dict[aKey] / total, 2) * 100, "%")
            print("*" * 80)
            instance += 1

        if key[0] == 'closest':
            y_pred.append(predicted_class_distance)

        if not predicted_class_distance and not predicted_class:
            for_app += 1

        if true_classification >= false_classification:
            tc += 1
            if tp_bool:
                avg = min_seq_length / timeseries.length()
                if avg > 1:
                    avg = 1
                avg_length += avg
        else:
            fc += 1

    app = (len(list_timeseries) - for_app) / float(len(list_timeseries))
    avg_length /= len(list_timeseries)
    # acc = float(tc) / float(tc + fc)
    sk_acc = sk_report = sk_acc_maj = sk_report_maj = 0
    if y_pred:
        sk_acc = accuracy_score(y_true, y_pred)
        sk_precision, sk_recall, sk_fscore, sk_support = precision_recall_fscore_support(y_true, y_pred,
                                                                                         average='macro')
        sk_report = classification_report(y_true, y_pred)
    if y_pred_maj:
        sk_acc_maj = accuracy_score(y_true, y_pred_maj)
        sk_precision_maj, sk_recall_maj, sk_fscore_maj, sk_support_maj = precision_recall_fscore_support(y_true,
                                                                                                         y_pred_maj,
                                                                                                         average=
                                                                                                         'macro')

    acc = sk_acc
    if sk_acc < sk_acc_maj:
        acc = sk_acc_maj
        sk_report = classification_report(y_true, y_pred_maj)

    return acc * 100, avg_length * 100, sk_acc * 100, sk_report, sk_acc_maj * 100, sk_report_maj, app * 100

#distance_measure: brute, mass_v1, mass_v2
#return: shap_found, min_dist
def pattern_found(a_timeseries, a_shapelet, distance_measure):
    if(distance_measure == "mass_v2" ):
        dist_profile = sm.mass_v2(a_timeseries, a_shapelet)
        if (min(dist_profile) <= a_shapelet.dist_threshold):
            pattern_found = True
        else:
            pattern_found = False
    return pattern_found
