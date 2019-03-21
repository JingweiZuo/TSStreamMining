import pandas as pd
from pandas.core import datetools
import numpy as np
from sklearn import model_selection, tree, metrics, neighbors, ensemble
from matplotlib import pyplot as plt
import tsfresh
import os
import xlrd
from math import ceil, floor
from tkinter import *
from sklearn.model_selection import train_test_split
from tsfresh.feature_extraction import MinimalFCParameters

class ml_methodes():

    #ds_Dir = '/Users/Jingwei/Desktop/Stage_DAVID/Master/Biblio_pour_Raef/CONVOIEMENTS/chocs_et_climat/2009_Quebec/aller/'
    #xlsfile = r'Classeur1_copy.xls'
    #book = xlrd.open_workbook(ds_Dir + xlsfile)

    #xlrd is used to get the names of all sheets in the book
    #count = len(book.sheets())
    #X_train, X_test, y_train, y_test = 0


    def __init__(self,dir,master):
        self.master = master
        from math import ceil, floor
        self.master_df = pd.DataFrame(columns=['number','time','x','y','z','sum','class'])

        book = pd.read_excel(dir,sheet_name = None, index=False, encoding='utf8',engine='xlrd')
        for sheet_key in book:
            df_sheet = book[sheet_key]
            df_sheet = df_sheet.rename(index=str, columns={'Reading Number': 'number', 'Date and Time (UTC+01:00)':'time', 'Shock - X Axis (g)':'x', 'Shock - Y Axis (g)':'y', 'Shock - Z Axis (g)':'z', 'Shock - Vector Sum (g)':'sum' })
            df_sheet = df_sheet.dropna(axis=0, how='any')
            # "master_df" is the original data, without segmentation of time series.
            self.master_df = self.master_df.append(df_sheet)
        '''
        self.direc = dir
        self.book = xlrd.open_workbook(self.direc)
        for sheet in self.book.sheets():
            print(sheet.name)

        df_Louvre_CDG = pd.read_excel(self.book,sheet_name='Louvre_CDG',index=False, encoding='utf8',engine='xlrd')
        df_rest = pd.read_excel(self.book,sheet_name='JFK-MNBAQ',index=False, encoding='utf8',engine='xlrd')
        #print(df_Louvre_CDG.head())
        df_Louvre_CDG = df_Louvre_CDG.rename(index=str, columns={'Reading Number': 'number', 'Date and Time (UTC+01:00)':'time', 'Shock - X Axis (g)':'x', 'Shock - Y Axis (g)':'y', 'Shock - Z Axis (g)':'z', 'Shock - Vector Sum (g)':'sum' })
        df_rest = df_rest.rename(index=str, columns={'Reading Number': 'number', 'Date and Time (UTC+01:00)':'time', 'Shock - X Axis (g)':'x', 'Shock - Y Axis (g)':'y', 'Shock - Z Axis (g)':'z', 'Shock - Vector Sum (g)':'sum' })

        print('df_Louvre_CDG.info_before\n')
        print(df_Louvre_CDG.info())
        print('df_rest.info_before\n')
        print(df_rest.info())
        #remove the lines which are not labeled
        df_Louvre_CDG = df_Louvre_CDG.dropna(axis=0, how='any')
        df_rest = df_rest.dropna(axis=0, how='any')
        self.master_df = self.master_df.append(df_Louvre_CDG)
        self.master_df = self.master_df.append(df_rest)
        '''
        self.classes = {
            1: 'Transportation by truck from Louvre to Air France Warehouse',
        # 20 Janv, 10:40-11:25, 21 Janv, 05:25-16:00
            2: 'Unloading the truck',  # 20 Janv, 11:25-13:30
            3: 'Transport in cargo area and loading of the plane',  # 20 Janv,16:48-16:55
            4: 'Takeoff',  # 20 Janv, 18:31-18:52
            5: 'In flight',  # 20 Janv, 18:52 - 21 Janv, 01:51
            6: 'Landing',  # 21 Janv, 01:51-02:36
            7: 'Unloading from the plane and loading to the truck',  # 21 Janv, 03:00-05:25
            8: 'No activity'
        }
        #the classes that we take into account
        self.class_name = [1, 2, 3, 4, 5, 6, 7, 8]
        '''
        #print(master_df.info())
        for i in self.classes.keys():
            len_data_i = len(self.master_df[self.master_df['class']==i])
            print(len_data_i)
            plt.figure()
            plt.xlim((0, len_data_i))
            plt.xlabel('instance number')
            plt.ylabel('g')
            #my_x_ticks = np.arange(0, len_data_i, 1000)
            #plt.xticks(my_x_ticks)
            self.master_df['x'].plot(legend='true')
            self.master_df['y'].plot(legend='true')
            self.master_df['z'].plot(legend='true')
            self.master_df['sum'].plot(legend='true')
            plt.title(self.classes.get(i))
        '''
    #Préparer des données afin de les mettre dans le format en entrée de tsfresh
    #Add 'id' for each sub-section in the input file
    def Preparation(self, df,id,y,length, class_flag):
        rdf = pd.DataFrame(columns=['time','x', 'y','z','sum','id'], dtype=float)
        if (class_flag =="with_class"):
            #preparation for the training file
            for c in self.class_name:
                #take the dataframe of class c
                df_class = df[df['class']==c]
                sLength = len(df_class['x'])
                df_class = df_class.drop(columns = ['number','class'])
                df_class = df_class.assign(id = pd.Series(0, index=df_class.index))
                #take the integer value in manner 'floor' of the lines' number
                num_sup = len(df_class) % length
                #there are 2 parts of data to add into 'rdf', the first 100*n lines, the 2nd num_sup lines
                df_i = np.array_split(df_class[:-num_sup],floor(len(df_class)/length))
                for df_i_i in df_i:
                    df_i_i['id'] = id
                    rdf = rdf.append(df_i_i,ignore_index = True)
                    y.append(c)
                    id +=1;
                #append the 2nd part in the list
                if(num_sup != 0):
                    df_i_i = df_class[-num_sup:-1].copy()
                    df_i_i['id'] = id
                    #df_i_i.is_copy = False
                    rdf = rdf.append(df_i_i,ignore_index = True)
                    '''
                    #can't assign directly the value to 'Series' like: df_class[-num_sup:-1]['id']= id 
                    df_class[-num_sup:-1] = df_class[-num_sup:-1].assign(id = pd.Series(id, index=df_class.index))
                    rdf.append(df_class[-num_sup:-1])
                    '''
                    y.append(c)
                    id+=1;
            return rdf, id, y
        else:
            #preparation for the test file
            df = df.drop(columns = ['number'])
            df = df.assign(id = pd.Series(0, index=df.index))
            #take the integer value in manner 'floor' of the lines' number
            num_sup = len(df) % length
            #there are 2 parts of data to add into 'rdf', the first 100*n lines, the 2nd num_sup lines
            df_i = np.array_split(df[:-num_sup],floor(len(df)/length))
            for df_i_i in df_i:
                df_i_i['id'] = id
                rdf = rdf.append(df_i_i,ignore_index = True)
                id +=1;
            #append the 2nd part in the list
            df_i_i = df[-num_sup:-1].copy()
            df_i_i['id'] = id
            #df_i_i.is_copy = False
            rdf = rdf.append(df_i_i,ignore_index = True)
            '''
            #can't assign directly the value to 'Series' like: df_class[-num_sup:-1]['id']= id 
            df_class[-num_sup:-1] = df_class[-num_sup:-1].assign(id = pd.Series(id, index=df_class.index))
            rdf.append(df_class[-num_sup:-1])
            '''
            id+=1;
            return rdf, id

    def ts_learn(self, flag_mode):
        if (flag_mode == "train"):
            rdf, id, y = self.Preparation(self.master_df,0,[],20, "with_class")
            '''
                Output the execution data to Text Window in the main page
            '''
            self.master.text_features.insert(END, "Information of input file: \n")
            self.master.text_features.insert(END, rdf.describe())

            print(rdf.describe())
            print('length of feature tables is :' + str(len(y)))

            X = tsfresh.extract_features(rdf, column_id='id', column_kind=None, column_value=None, column_sort='time', default_fc_parameters= MinimalFCParameters())
            '''
                Output the execution data to Text Window in the main page
            '''
            self.master.text_features.insert(END,"\nThe features extracted are:\n")
            self.master.text_features.insert(END, ''.join(str(x )+ '  ' for x in X.head().columns.get_values().tolist()))

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y[:len(X)], test_size=0.2, random_state=0)
            #np.any(np.isnan(X_train))
            #np.all(np.isfinite(X_train))
            self.X_train = self.X_train.dropna(axis=0, how='any')
            np.any(np.isnan(self.X_train))
            X.to_csv(r'/Users/Jingwei/Desktop/Feature_C2RMF.csv',mode='a',index='false')
            print('The size (number of lines) of feature table is ' + str(len(X)))
        else:
            if(flag_mode == "test"):
                rdf, id = self.Preparation(self.test_df, 0, [], 20, "without_class")
                self.X = tsfresh.extract_features(rdf, column_id='id', column_kind=None, column_value=None, column_sort='time', default_fc_parameters= MinimalFCParameters())
                self.X = self.X.dropna(axis=0, how='any')
                np.any(np.isnan(self.X))

    def train_model(self, classifier):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

        train_method = KNeighborsClassifier(n_neighbors=5)
        if(classifier=="KNeighbors"):
            #KNearestNeighborsClassifier
            train_method = KNeighborsClassifier(n_neighbors=5)
        else:
            if(classifier=="Decision tree"):
                train_method = DecisionTreeClassifier()
            else:
                if(classifier=="RandomForest"):
                    train_method = RandomForestClassifier()
                else:
                    if(classifier=="AdaBoost"):
                        train_method = AdaBoostClassifier()
                    else:
                        if(classifier=="GradientBoosting"):
                            train_method = GradientBoostingClassifier()

        train_method = train_method.fit(self.X_train, self.y_train)
        self.train_m = train_method
        y_predict = train_method.predict(self.X_test)
        #output the evaluation
        self.master.text_features.insert(END, "\n\nEvaluation result of learning model:")
        self.master.text_features.insert(END, '\nAccuracy: ' + str(accuracy_score(self.y_test, y_predict)))
        self.master.text_features.insert(END, '\nConfusion matrix :\n' + str(confusion_matrix(self.y_test,y_predict)))
        self.master.text_features.insert(END, '\nclassification report:\n' + str(classification_report(self.y_test, y_predict)))

    def predict_class(self, file):
        #book = xlrd.open_workbook(file)
        book = pd.read_excel(file,sheet_name = None, index=False, encoding='utf8',engine='xlrd')
        self.test_df = pd.DataFrame(columns=['number','time','x','y','z','sum'])

        for sheet_key in book:
            print(book[sheet_key].head())
            df_sheet = book[sheet_key]
            df_sheet = df_sheet.rename(index=str, columns={'Reading Number': 'number', 'Date and Time (UTC+01:00)':'time', 'Shock - X Axis (g)':'x', 'Shock - Y Axis (g)':'y', 'Shock - Z Axis (g)':'z', 'Shock - Vector Sum (g)':'sum' })
            df_sheet = df_sheet.dropna(axis=0, how='any')
            # "master_df" is the original data, without segmentation of time series.
            self.test_df = self.test_df.append(df_sheet)

        # Step 1: segment and extract features the input testing file
        self.ts_learn("test")
        # "y_predict" is the prediction result of input testing file, for each segmentation
        y_predict = self.train_m.predict(self.X)
        return y_predict
