import numpy as np
import pandas as pd

from numpy import array

from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential

from keras.callbacks import History, EarlyStopping, Callback

from keras.layers import Dense, LSTM, Activation, Dropout, GRU, RepeatVector, GaussianNoise, TimeDistributed, Flatten, BatchNormalization
from keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences
import keras
from keras.models import load_model

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.utils import plot_model

import math
from sklearn.metrics import mean_squared_error







class common:
    
    METRICS = []
    JOBS = []
    URL_PATTERN = ""
    DICT_ALL_CSV = {}
    
    @staticmethod
    def init():
        from keras import backend as K
        import tensorflow as tf
        import random as rn
        import os
        
        os.environ['PYTHONHASHSEED'] = '0'

        # Setting the seed for numpy-generated random numbers
        np.random.seed(123)

        # Setting the seed for python random numbers
        rn.seed(123)

        # Setting the graph-level random seed.
        tf.set_random_seed(123)

        sess = tf.Session(graph=tf.get_default_graph())
        K.set_session(sess)
        
    @staticmethod
    def set_url_pattern(url_pattern):
        assert "JOB_ID" in URL_PATTERN and "FREQ" in URL_PATTERN and "METRIC" in URL_PATTERN, "El patron introducido no es correcto"
        
        common.URL_PATTERN = url_pattern
        
    @staticmethod
    def __generate_series__(dataset, output, n_input=1, batch_size=1):
	    return TimeseriesGenerator(dataset, output, length=n_input, batch_size=batch_size)
    
    @staticmethod
    def __df_padding__(tam, fill_with=0):
        df_nan = pd.DataFrame([[fill_with]*len(common.METRICS)]*tam)
        df_nan.columns = common.METRICS
    
        return df_nan

    @staticmethod
    def read_csv(job_id, metric, frequency="5", header=0, index_col=0):
        import os 

        path = common.URL_PATTERN.replace("JOB_ID", job_id).replace("FREQ",frequency).replace("METRIC", metric)
        
        assert os.path.isfile(path), "El fichero correspondiente al trabajo no existe"
        
        return pd.read_csv(path, header=header, index_col=index_col)
        
    @staticmethod
    def get_nodos_by_job(job_id, frequency="5"):
        return list(common.read_csv(job_id, common.METRICS[0]))
    
    #Esto lo hago por que se me dio el caso de que para el mismo trabajo, mismo nodo, hay metricas con distintos intervalos. 
    @staticmethod
    def __fix_error_long_ts__(dict_res, nodo):
        import sys
        
        dict_df_temp = {}

        len_min_ts = sys.maxsize

        for key, value in dict_res.items():
            dict_df_temp[key] = value[nodo].values
            len_min_ts = min(len_min_ts, len(dict_df_temp[key]))

        for key, value in dict_df_temp.items():
            dict_df_temp[key] = value[:len_min_ts]
        
        return dict_df_temp
        
    @staticmethod
    def get_df_nodo(job_id, nodo, frequency="5"):
        return pd.DataFrame.from_dict(common.__fix_error_long_ts__(common.DICT_ALL_CSV[job_id], nodo))
    
    
    @staticmethod
    def read_all_csv(frequency="5"):
        for job_id in common.JOBS:
            common.DICT_ALL_CSV[job_id] = {}
            for metric in common.METRICS:
                common.DICT_ALL_CSV[job_id][metric] = common.read_csv(job_id, metric, frequency=frequency)
    
    
    @staticmethod
    def save_model(model, filename):
        import os
        path = os.path.realpath("Models/"+filename+'.h5')
        model.save(path)

    @staticmethod
    def load_model(filename):
        import os
        path = os.path.realpath("Models/"+filename+'.h5')
        assert os.path.isfile(path), "El path h5 correspondiente al modelo no existe"
        return load_model(path)
        
        
    @staticmethod
    def build_model_name(tipo, algoritmo, timesteps, epochs, n_neuronas, shuffle, callback):
        
        assert tipo in ["tb", "tob"], "El tipo es incorrecto"
        
        assert algoritmo in ["lstm_autoencoder_simple", "lstm_simple", "lstm_stacked", 
                            "lstm_autoencoder_stacked"], "El nombre del algoritmo es incorrecto"
        
        
        return "model_"+tipo+"_"+algoritmo+"_"+timesteps+"_"+epochs+"_"+n_neuronas+"_"+shuffle+"_"+callback
        
class train_on_batch:
    
    JOB_TEST = 20
    LOOK_BACK = 1
    scaler = None
    
    @staticmethod
    def all_df(frequency="5"):
        res = {}

        for job_id in common.JOBS:
            nodos = list(common.DICT_ALL_CSV[job_id][common.METRICS[0]])
            
            res[job_id] = {}
            
            
            for nodo in nodos:
                assert nodo in common.DICT_ALL_CSV[job_id][common.METRICS[0]], "El nodo no es valido"
                
                res[job_id][nodo] = common.get_df_nodo(job_id, nodo, frequency)

        return res
    
    @staticmethod
    def get_sequences_metric(output, dict_res, array_size):
        
        test_size = sum(array_size[-train_on_batch.JOB_TEST:])

        train_on_batch.scaler = MinMaxScaler(feature_range=(0, 1))
        dict_res_scaled = train_on_batch.scaler.fit_transform(dict_res)
        
        train_size = int(len(dict_res_scaled) - test_size)
        test_size = len(dict_res_scaled) - train_size
        trainX, testX = dict_res_scaled[0:train_size,:], dict_res_scaled[train_size:len(dict_res_scaled),:]    

        #Si la variable que se quiere como salida no existe, devuelve false
        if output not in dict_res: return False

        pos_output = list(dict_res).index(output)

        trainY = array([x[pos_output] for x in trainX])
        testY = array([x[pos_output] for x in testX])

        posIni = 0
        posFin = 0
        train_X, train_y, test_X, test_y = [],[],[],[]
        new_array_size = []
        for pos, batch_size in enumerate(array_size[:-train_on_batch.JOB_TEST]):

            posFin += batch_size
            train_gen = common.__generate_series__(array(trainX[posIni:posFin]), array(trainY[posIni:posFin]), train_on_batch.LOOK_BACK)

            new_array_size.append(len(train_gen))
            r = [x[0][0] for x in train_gen]

            train_X.extend([x[0][0] for x in train_gen])
            train_y.extend([x[1][0] for x in train_gen])
            posIni+=batch_size

        posIni = 0
        posFin = 0
        for batch_size in array_size[-train_on_batch.JOB_TEST:]:
            posFin += batch_size
            test_gen = common.__generate_series__(array(testX[posIni:posFin]), array(testY[posIni:posFin]), train_on_batch.LOOK_BACK)
            new_array_size.append(len(test_gen))
            test_X.extend([x[0][0] for x in test_gen])
            test_y.extend([x[1][0] for x in test_gen])
            posIni+=batch_size

        return [array(train_X), array(train_y), array(test_X), array(test_y), dict_res[output], new_array_size]

    
    @staticmethod
    def all_ts(metric_to_predict, frequency="5"):
        import random
        
        dict_all_df = train_basic.all_df(frequency=frequency)
        
        #Diccionario de series temporales por metrica
        dict_ts = {}

        df_concatenated = pd.DataFrame()
        
        combinating_ts = []
        for job_id, nodos in dict_all_df.items():
            for nodo, df in nodos.items():
                combinating_ts.append(dict_all_df[job_id][nodo])
        
        random.shuffle(combinating_ts)
        
        for ts in combinating_ts:
            df_concatenated = pd.concat([df_concatenated, ts])

        df_concatenated.reset_index(drop=True, inplace=True)
        
        temp = train_basic.get_sequences_job_metric(df_concatenated, metric_to_predict)

        dict_ts["trainX"] = temp[0]
        dict_ts["trainY"] =temp[1]
        dict_ts["testX"] = temp[2]
        dict_ts["testY"] = temp[3]
        dict_ts["output"] = temp[4]

        return dict_ts
    
    @staticmethod
    def all_ts(metric_to_predict, frequency="5"):
        import random
        
        dict_all_df = train_on_batch.all_df(frequency=frequency)
        
        #Diccionario de series temporales por metrica
        dict_ts = {}

        df_concatenated = pd.DataFrame()

        array_size = []

        combinating_ts = []
        for job_id, nodos in dict_all_df.items():
            for nodo, df in nodos.items():
                combinating_ts.append(dict_all_df[job_id][nodo])
                
        random.shuffle(combinating_ts)
        
        for ts in combinating_ts:
            df_concatenated = pd.concat([df_concatenated, ts])
            df_concatenated = pd.concat([df_concatenated, common.__df_padding__(5)])
            array_size.append(len(ts)+5)

        df_concatenated.reset_index(drop=True, inplace=True)
        
        temp = train_on_batch.get_sequences_metric(metric_to_predict, df_concatenated, array_size)

        dict_ts["trainX"] = temp[0]
        dict_ts["trainY"] =temp[1]
        dict_ts["testX"] = temp[2]
        dict_ts["testY"] = temp[3]
        dict_ts["output"] = temp[4]

        return dict_ts, temp[5]
    
    @staticmethod
    def test_hiperparameters(dict_ts, array_size, tipo, arr_EPOCHS, n_neuronas):
        for activation in ["relu","tanh"]:
            for optimizer in ["adam","rmsprop"]:
                for EPOCHS in arr_EPOCHS:
                    for n_neur in n_neuronas:
                        if tipo=="stacked":
                            train_on_batch.train_on_batch_lstm_stacked(dict_ts, n_neur, EPOCHS, array_size, activation, optimizer)
                        elif tipo=="simple":
                            train_on_batch.train_on_batch_lstm_simple(dict_ts, n_neur, EPOCHS, array_size, activation, optimizer)
                        elif tipo=="autoencoder":
                            train_on_batch.train_on_batch_lstm_autoencoder(dict_ts, n_neur, EPOCHS, array_size, activation, optimizer)
    
    @staticmethod
    def train_on_batch_lstm_stacked(dict_ts, n_neuronas, EPOCHS, array_size, activation='relu', optimizer="adam"):
        
        keras.backend.clear_session()
        
        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]
        
        n_layers = len(n_neuronas)
        assert n_layers>1, "El numero de capas debe ser mayor a 1"
        
        # design network
        model = Sequential()
        model.add(LSTM(n_neuronas[0], activation=activation,
                       input_shape =(train_X.shape[1], train_X.shape[2]),
                       return_sequences=True))
        
        model.add(Dropout(0.2))
        
        for i in range(1, n_layers-1):
            model.add(LSTM(n_neuronas[i],return_sequences=True))
            model.add(Dropout(0.2))
        
        model.add(LSTM(n_neuronas[n_layers-1]))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation="linear"))

        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'acc'])
        
        model = train_on_batch.train(model, train_X, train_y, n_neuronas, EPOCHS, array_size, activation, optimizer)
    
        return {"model":model, "train_X":train_X, "train_y":train_y, "test_X":test_X, 
                "test_y":test_y, "output":dict_ts["output"], "array_size":array_size}
    
    
    @staticmethod
    def train_on_batch_lstm_simple(dict_ts, n_neuronas, EPOCHS, array_size, activation='relu', optimizer="adam"):
        
        keras.backend.clear_session()
        
        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]

        # design network
        model = Sequential()
        model.add(LSTM(n_neuronas[0], activation=activation,
                       input_shape =(train_X.shape[1], train_X.shape[2])))
        
        model.add(Dense(1, activation="linear"))

        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'acc'])
        
        model = train_on_batch.train(model, train_X, train_y, n_neuronas, EPOCHS, array_size, activation, optimizer)
    
        return {"model":model, "train_X":train_X, "train_y":train_y, "test_X":test_X, 
                "test_y":test_y, "output":dict_ts["output"], "array_size":array_size}
    
    @staticmethod
    def train_on_batch_lstm_autoencoder_simple(dict_ts, n_neuronas, EPOCHS, array_size, activation='relu', optimizer="adam"):
        
        keras.backend.clear_session()
        
        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]

        # design network
        model = Sequential()
        model.add(GaussianNoise(0.01, batch_input_shape=(train_basic.BATCH_SIZE, train_X.shape[1], train_X.shape[2])))
        model.add(LSTM(n_neuronas[0], activation=activation, input_shape =(train_X.shape[1], train_X.shape[2]), name="encoder"))
        model.add(RepeatVector(train_X.shape[1]))
        model.add(LSTM(n_neuronas[0], activation='relu', return_sequences=True, name="decoder"))
        model.add(Flatten())
        model.add(Dense(1, activation="linear"))

        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'acc'])
        
        model = train_on_batch.train(model, train_X, train_y, n_neuronas, EPOCHS, array_size, activation, optimizer)
        
        return {"model":model, "train_X":train_X, "train_y":train_y, "test_X":test_X, 
                "test_y":test_y, "output":dict_ts["output"], "array_size":array_size}
    
    @staticmethod
    def train(model, train_X, train_y, n_neuronas, EPOCHS, array_size, activation, optimizer):
        array_size_train = array_size[:-train_on_batch.JOB_TEST]
        array_mse_train, array_mse_test, array_acc_train, array_acc_test, array_mae_train, array_mae_test = [], [], [], [], [], []
        for epoch in range(EPOCHS):
            posIni, posFin, temp, temp2 = 0, 0, 0, 0
            for batch_size in array_size_train:
                posFin += batch_size
                X, Y = train_X[posIni:posFin], train_y[posIni:posFin]
                temp = model.train_on_batch(X,Y)
                temp2 = model.test_on_batch(X,Y)
                model.reset_states()
                posIni = posFin
            array_mse_train.append(temp[0])
            array_mse_test.append(temp2[0])
            array_acc_train.append(temp[2])
            array_acc_test.append(temp2[2])
            array_mae_train.append(temp[1])
            array_mae_test.append(temp2[1])
        
        print("MAE TRAIN: ",min(array_mae_train), " MAE TEST: ",min(array_mae_test))
        plt.plot(range(EPOCHS), array_mae_train, label='Train_on_batch mae')
        plt.plot(range(EPOCHS), array_mae_test, label='Test_on_batch mae')
        plt.gcf().set_size_inches(11,8)
        plt.legend(framealpha=1, frameon=True);
        plt.title(("Variaciones (EPOCHS=",EPOCHS," n_neuronas=",n_neuronas," activation=",activation," optimizer=",optimizer))
        plt.show()
    
        return model
    
    
    @staticmethod
    def predict_on_batch(dict_train):
        
        model,train_X,train_y, test_X = dict_train["model"], dict_train["train_X"], dict_train["train_y"], dict_train["test_X"]
        test_y, out_real, array_size = dict_train["test_y"], dict_train["output"], dict_train["array_size"]
        
        array_size_train = array_size[:-train_on_batch.JOB_TEST]
        array_size_test = array_size[-train_on_batch.JOB_TEST:]
        
        predict_train = []
        train_real = []
        predict_test = []
        test_real = []
        
        scaler_tmp = MinMaxScaler(feature_range=(0, 1))
        scaler_tmp.fit_transform(array(out_real).reshape(-1, 1))
         
        posIni, posFin = 0, 0
        for batch_size in array_size_train:
            posFin += batch_size
            X, Y = train_X[posIni:posFin], train_y[posIni:posFin]
            predict_train = model.predict_on_batch(X)
            
            rmse = math.sqrt(mean_squared_error(Y, predict_train))
            print('train RMSE: %.3f' % rmse)
            
            predict_train_real = scaler_tmp.inverse_transform(predict_train)
            train_y_real = scaler_tmp.inverse_transform(Y.reshape((len(Y), 1)))
            rmse = math.sqrt(mean_squared_error(train_y_real, predict_train_real))
            print('train RMSE: %.3f' % rmse)
            
            plt.plot(np.concatenate((predict_train_real)), label='Estimacion') 
            plt.plot(np.concatenate((train_y_real)), label='Real', alpha=0.5)
            plt.legend(framealpha=1, frameon=True);
            plt.gcf().set_size_inches(11,8)
            plt.title("Gráfica no escalada TRAIN")
            plt.show()
        
            posIni = posFin
        
        print("TEEEEEESSSSSSTTTT")
        posIni, posFin = 0, 0
        for batch_size in array_size_test:
            posFin += batch_size
            X, Y = test_X[posIni:posFin], test_y[posIni:posFin]
            predict_test = model.predict_on_batch(X)
            
            rmse = math.sqrt(mean_squared_error(Y, predict_test))
            print('test RMSE: %.3f' % rmse)
            
            predict_test_real = scaler_tmp.inverse_transform(predict_test)
            test_y_real = scaler_tmp.inverse_transform(Y.reshape((len(Y), 1)))
            
            # calculate RMSE
            rmse = math.sqrt(mean_squared_error(test_y_real, predict_test_real))
            print('test RMSE: %.3f' % rmse)
            
            plt.plot(np.concatenate((predict_test_real)), label='Estimacion') 
            plt.plot(np.concatenate((test_y_real)), label='Real', alpha=0.5)
            plt.legend(framealpha=1, frameon=True);
            plt.gcf().set_size_inches(11,8)
            plt.title("Gráfica no escalada TEST")
            plt.show()
            
            posIni = posFin

        
class train_basic:
    
    JOB_TEST = 20
    LOOK_BACK = 1
    scaler = None
    BATCH_SIZE = 0
    
    @staticmethod
    def check_all_df_same_tam(dict_all_df, long_ts_longer):
        res = True

        for job_id, nodos in dict_all_df.items():
            for nodo, df in nodos.items():
                res = res and len(dict_all_df[job_id][nodo]) == long_ts_longer

        return res
    
    @staticmethod
    def pad_all_df(dict_all_df, long_ts_longer):
        for job_id, nodos in dict_all_df.items():
            for nodo, df in nodos.items():
                if len(dict_all_df[job_id][nodo]) < long_ts_longer:
                    dif = long_ts_longer - len(dict_all_df[job_id][nodo])
                    dict_all_df[job_id][nodo] = pd.concat([dict_all_df[job_id][nodo], common.__df_padding__(dif)])
                else:
                    break
                    
        assert train_basic.check_all_df_same_tam(dict_all_df, long_ts_longer), "La longitud de todos los df no son del mismo tam"
        
        train_basic.BATCH_SIZE = long_ts_longer
        
        return dict_all_df
    
    @staticmethod
    def all_df(frequency="5"):
        res = {}

        #Variable que indica la longitud de la serie más larga
        long_ts_longer = 0

        for job_id in common.JOBS:
            nodos = list(common.DICT_ALL_CSV[job_id][common.METRICS[0]])
            
            res[job_id] = {}

            for nodo in nodos:
                assert nodo in common.DICT_ALL_CSV[job_id][common.METRICS[0]], "El nodo no es valido"
                
                res[job_id][nodo] = common.get_df_nodo(job_id, nodo, frequency)

            long_ts_longer = max(long_ts_longer, len(res[job_id][nodo]))
        
        res = train_basic.pad_all_df(res, long_ts_longer)
        
        return res
    
    @staticmethod
    def get_sequences_job_metric(dict_res, metric_to_predict):

        train_basic.scaler = MinMaxScaler(feature_range=(0, 1))
        dict_res_scaled = train_basic.scaler.fit_transform(dict_res)

        train_size = int(len(dict_res_scaled) - train_basic.BATCH_SIZE*train_basic.JOB_TEST)
        test_size = len(dict_res_scaled) - train_size
        trainX, testX = dict_res_scaled[0:train_size,:], dict_res_scaled[train_size:len(dict_res_scaled),:]    

        #Si la variable que se quiere como salida no existe, devuelve false
        if metric_to_predict not in dict_res: return False

        pos_output = list(dict_res).index(metric_to_predict)

        trainY = array([x[pos_output] for x in trainX])
        testY = array([x[pos_output] for x in testX])
        
        posIni = 0
        posFin = 0
        train_X, train_y, test_X, test_y = [],[],[],[]
        
        #print("len trainX: ",len(trainX)," % ",len(trainX)%train_basic.BATCH_SIZE, " / ",int(len(trainX)/train_basic.BATCH_SIZE))
        for i in range(0,int(len(trainX)/train_basic.BATCH_SIZE)):

            posFin += train_basic.BATCH_SIZE
            
            train_gen = common.__generate_series__(array(trainX[posIni:posFin]), array(trainY[posIni:posFin]), train_basic.LOOK_BACK)
            
            train_X.extend([x[0][0] for x in train_gen])
            train_y.extend([x[1][0] for x in train_gen])
            
            train_X.extend([x for x in train_X[-train_basic.LOOK_BACK:]])
            train_y.extend([y for y in train_y[-train_basic.LOOK_BACK:]])
            
            posIni=posFin
        
        #print("len train_X: ",len(train_X)," % ",len(train_X)%train_basic.BATCH_SIZE)
        
        posIni = 0
        posFin = 0
        for i in range(0,int(len(testX)/train_basic.BATCH_SIZE)):
            
            posFin += train_basic.BATCH_SIZE
            test_gen = common.__generate_series__(array(testX[posIni:posFin]), array(testY[posIni:posFin]), train_basic.LOOK_BACK)

            test_X.extend([x[0][0] for x in test_gen])
            test_y.extend([x[1][0] for x in test_gen])
            
            test_X.extend([x for x in test_X[-train_basic.LOOK_BACK:]])
            test_y.extend([y for y in test_y[-train_basic.LOOK_BACK:]])
            
            posIni=posFin

        return array(train_X), array(train_y), array(test_X), array(test_y), dict_res[metric_to_predict]

    @staticmethod
    def all_ts(metric_to_predict, frequency="5"):
        import random
        
        dict_all_df = train_basic.all_df(frequency=frequency)
        
        #Diccionario de series temporales por metrica
        dict_ts = {}

        df_concatenated = pd.DataFrame()
        
        combinating_ts = []
        for job_id, nodos in dict_all_df.items():
            for nodo, df in nodos.items():
                combinating_ts.append(dict_all_df[job_id][nodo])
        
        random.shuffle(combinating_ts)
        
        for ts in combinating_ts:
            df_concatenated = pd.concat([df_concatenated, ts])

        df_concatenated.reset_index(drop=True, inplace=True)
        
        temp = train_basic.get_sequences_job_metric(df_concatenated, metric_to_predict)

        dict_ts["trainX"] = temp[0]
        dict_ts["trainY"] =temp[1]
        dict_ts["testX"] = temp[2]
        dict_ts["testY"] = temp[3]
        dict_ts["output"] = temp[4]

        return dict_ts

    
    @staticmethod
    def train(model, train_X, train_y, test_X, test_y, n_neuronas, EPOCHS, activation, optimizer, callbacks=None):
        
        model.fit(train_X, train_y, epochs=EPOCHS, batch_size=train_basic.BATCH_SIZE, verbose=2, shuffle=False, callbacks=callbacks,
                  validation_split=0.25)

        return model
    
    
    @staticmethod
    def train2(model, train_X, train_y, test_X, test_y, n_neuronas, EPOCHS, activation, optimizer, callbacks=None):
        
        array_mae_train = []
        for i in range(EPOCHS):
            model.fit(train_X, train_y, epochs=1, batch_size=train_basic.BATCH_SIZE, verbose=2, shuffle=False, callbacks=callbacks,
                     validation_split=0.25)
            
            array_mae_train.append(model.evaluate(train_X, train_y, batch_size=train_basic.BATCH_SIZE, verbose=0))
        
        print("MSE TRAIN: ",min(array_mae_train))
        plt.plot(range(EPOCHS)[20:], array_mae_train[20:])
        plt.gcf().set_size_inches(11,8)
        plt.title(("Variaciones (EPOCHS=",EPOCHS," n_neuronas=",n_neuronas," activation=",activation," optimizer=",optimizer))
        plt.show()
        
        return model
    
    @staticmethod
    def train_lstm_simple(dict_ts, n_neuronas, EPOCHS,activation='relu', optimizer="adam", callback=False):
        keras.backend.clear_session()

        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]
        
        cbs = [History(), EarlyStopping(monitor='loss', patience=10, min_delta=0.0003, verbose=0)] if callback else None
        
        # design network
        model = Sequential()
        model.add(LSTM(n_neuronas[0], input_shape=(train_X.shape[1], train_X.shape[2]), activation=activation))
        
        model.add(Dense(1, activation="linear"))

        model.compile(loss='mae', optimizer=optimizer)
        
        model = train_basic.train(model, train_X, train_y, test_X, test_y, n_neuronas[0], EPOCHS, activation, optimizer, cbs)

        return {"model":model, "train_X":train_X, "train_y":train_y, "test_X":test_X, 
                "test_y":test_y, "output":dict_ts["output"]}
    
    
    @staticmethod
    def train_lstm_stacked_dropout(dict_ts, n_neuronas, EPOCHS,activation='relu', optimizer="adam", callback=False):
        keras.backend.clear_session()
        
        n_layers = len(n_neuronas)
        assert n_layers>1, "El numero de capas debe ser mayor a 1"
        
        cbs = [History(), EarlyStopping(monitor='loss', patience=10, min_delta=0.0003, verbose=0)] if callback else None
        
        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]

        model = Sequential()
        model.add(LSTM(n_neuronas[0], activation=activation, return_sequences=True,
                      input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(0.2))
        
        for i in range(1, n_layers-1):
            model.add(LSTM(n_neuronas[i],return_sequences=True))
            model.add(Dropout(0.2))
        
        model.add(LSTM(n_neuronas[n_layers-1]))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation="linear"))

        model.compile(loss='mae', optimizer=optimizer)

        model = train_basic.train(model, train_X, train_y, test_X, test_y, n_neuronas[0], EPOCHS, activation, optimizer, cbs)
        
        return {"model":model, "train_X":train_X, "train_y":train_y, "test_X":test_X, 
                "test_y":test_y, "output":dict_ts["output"]}
    
    @staticmethod
    def train_lstm_stacked_batch_normalization(dict_ts, n_neuronas, EPOCHS,activation='relu', optimizer="adam", callback=False):
        keras.backend.clear_session()
        
        n_layers = len(n_neuronas)
        assert n_layers>1, "El numero de capas debe ser mayor a 1"
        
        cbs = [History(), EarlyStopping(monitor='loss', patience=10, min_delta=0.0003, verbose=0)] if callback else None
        
        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]

        model = Sequential()
        model.add(LSTM(n_neuronas[0], activation=activation, return_sequences=True, use_bias=False,
                       input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(BatchNormalization())
        for i in range(1, n_layers-1):
            model.add(LSTM(n_neuronas[i],return_sequences=True, use_bias=False))
            model.add(BatchNormalization())
        
        model.add(LSTM(n_neuronas[n_layers-1], use_bias=False))
        model.add(BatchNormalization())

        model.add(Dense(1, activation="linear"))

        model.compile(loss='mae', optimizer=optimizer)

        model = train_basic.train(model, train_X, train_y, test_X, test_y, n_neuronas[0], EPOCHS, activation, optimizer, cbs)
        
        return {"model":model, "train_X":train_X, "train_y":train_y, "test_X":test_X, 
                "test_y":test_y, "output":dict_ts["output"]}
    
    
    @staticmethod
    def train_lstm_autoencoder_simple(dict_ts, n_neuronas, EPOCHS,activation='relu', optimizer="adam", callback=False):
        keras.backend.clear_session()

        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]
        
        cbs = [History(), EarlyStopping(monitor='loss', patience=10, min_delta=0.0003, verbose=0)] if callback else None
        
        # design network
        model = Sequential()
        model.add(GaussianNoise(0.01, input_shape=(train_X.shape[1], train_X.shape[2])))
        
        model.add(LSTM(n_neuronas[0], activation=activation, name="encoder"))
        model.add(RepeatVector(train_X.shape[1]))
        model.add(LSTM(n_neuronas[0], activation='relu', return_sequences=True, name="decoder"))
        
        model.add(Flatten())
        model.add(Dense(1, activation="linear"))
        #model.add(TimeDistributed(Dense(...)))
        model.compile(loss='mae', optimizer=optimizer)
        
        model = train_basic.train(model, train_X, train_y, test_X, test_y, n_neuronas[0], EPOCHS, activation, optimizer, cbs)
    
        
        return {"model":model, "train_X":train_X, "train_y":train_y, "test_X":test_X, 
                "test_y":test_y, "output":dict_ts["output"]}
    
    
    @staticmethod
    def train_lstm_autoencoder_stacked(dict_ts, n_neuronas, EPOCHS,activation='relu', optimizer="adam", callback=False):
        keras.backend.clear_session()

        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]
        
        cbs = [History(), EarlyStopping(monitor='loss', patience=10, min_delta=0.0003, verbose=0)] if callback else None
        
        # design network
        model = Sequential()
        model.add(LSTM(n_neuronas[0], input_shape=(train_X.shape[1], train_X.shape[2]), activation=activation, 
                       name="encoder"))
        model.add(RepeatVector(train_X.shape[1]))
        model.add(LSTM(n_neuronas[0], activation=activation))
        model.add(RepeatVector(train_X.shape[1]))
        model.add(LSTM(n_neuronas[0], activation=activation, return_sequences=True, name="decoder"))
        model.add(Flatten())
        model.add(Dense(1, activation="linear"))

        model.compile(loss='mae', optimizer=optimizer)
        
        model = train_basic.train(model, train_X, train_y, test_X, test_y, n_neuronas[0], EPOCHS, activation, optimizer,cbs)
        
        return {"model":model, "train_X":train_X, "train_y":train_y, "test_X":test_X, 
                "test_y":test_y, "output":dict_ts["output"]}

    @staticmethod
    def predict(dict_train):
        
        model,train_X,train_y, test_X = dict_train["model"], dict_train["train_X"], dict_train["train_y"], dict_train["test_X"]
        test_y, out_real = dict_train["test_y"], dict_train["output"]
        
        predict_train, train_real, predict_test, test_real = [], [], [], []
 
        scaler_tmp = MinMaxScaler(feature_range=(0, 1))
        scaler_tmp.fit_transform(array(out_real).reshape(-1, 1))
         
        posIni, posFin = 0, 0
        for i in range(0,int(len(train_X)/train_basic.BATCH_SIZE)):
            posFin += train_basic.BATCH_SIZE
            
            X, Y = train_X[posIni:posFin], train_y[posIni:posFin]
            predict_train = model.predict(X,batch_size=1)
            
            rmse = math.sqrt(mean_squared_error(Y, predict_train))
            print('train RMSE: %.3f' % rmse)
            
            
            predict_train_real = scaler_tmp.inverse_transform(predict_train)
            train_y_real = scaler_tmp.inverse_transform(Y.reshape((len(Y), 1)))
            rmse = math.sqrt(mean_squared_error(train_y_real, predict_train_real))
            print('train RMSE: %.3f' % rmse)
            
            plt.plot(np.concatenate((predict_train_real)), label='Estimacion') 
            plt.plot(np.concatenate((train_y_real)), label='Real', alpha=0.5)
            plt.legend(framealpha=1, frameon=True);
            plt.gcf().set_size_inches(11,8)
            plt.title("Gráfica no escalada TRAIN")
            plt.show()
            
            posIni=posFin
        
        posIni, posFin = 0, 0
        for i in range(0,int(len(test_X)/train_basic.BATCH_SIZE)):
            posFin += train_basic.BATCH_SIZE
            
            X, Y = test_X[posIni:posFin], test_y[posIni:posFin]
            predict_test = model.predict(X,batch_size=1)
            
            rmse = math.sqrt(mean_squared_error(Y, predict_test))
            print('train RMSE: %.3f' % rmse)
            
            
            predict_test_real = scaler_tmp.inverse_transform(predict_test)
            test_y_real = scaler_tmp.inverse_transform(Y.reshape((len(Y), 1)))
            rmse = math.sqrt(mean_squared_error(test_y_real, predict_test_real))
            print('train RMSE: %.3f' % rmse)
            
            plt.plot(np.concatenate((predict_test_real)), label='Estimacion') 
            plt.plot(np.concatenate((test_y_real)), label='Real', alpha=0.5)
            plt.legend(framealpha=1, frameon=True);
            plt.gcf().set_size_inches(11,8)
            plt.title("Gráfica no escalada TEST")
            plt.show()
            
            posIni=posFin
            
            
class train_avanzado:
    
    
    @staticmethod
    def train_lstm_avanzado_1(dict_ts, n_neuronas, EPOCHS,activation='relu', optimizer="adam", callback=False):
        keras.backend.clear_session()
        
        n_layers = len(n_neuronas)
        assert n_layers>1, "El numero de capas debe ser mayor a 1"
        
        cbs = [History(), EarlyStopping(monitor='loss', patience=10, min_delta=0.0003, verbose=0)] if callback else None
        
        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]

        model = Sequential()
        model.add(GaussianNoise(0.01, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(LSTM(n_neuronas[0], activation=activation, return_sequences=True, use_bias=False))
        model.add(BatchNormalization())
        for i in range(1, n_layers-1):
            model.add(LSTM(n_neuronas[i],return_sequences=True, use_bias=False))
            model.add(BatchNormalization())
        
        model.add(LSTM(n_neuronas[n_layers-1], use_bias=False))
        model.add(BatchNormalization())

        model.add(Dense(1, activation="linear"))

        model.compile(loss='mae', optimizer=optimizer)

        model = train_basic.train(model, train_X, train_y, test_X, test_y, n_neuronas[0], EPOCHS, activation, optimizer, cbs)
        
        return {"model":model, "train_X":train_X, "train_y":train_y, "test_X":test_X, 
                "test_y":test_y, "output":dict_ts["output"]}
    
    
    #PROBAR, YA QUE TIMEDISTRIBUTED NO ME FUNCIONA
    @staticmethod
    def train_lstm_autoencoder_avanzado(dict_ts, n_neuronas, EPOCHS,activation='relu', optimizer="adam", callback=False):
        keras.backend.clear_session()
        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]
        
        cbs = [History(), EarlyStopping(monitor='loss', patience=35, min_delta=0.0002, verbose=0,
                                       restore_best_weights=True)] if callback else None
        
        print(train_X.shape)
        # design network
        model = Sequential()
        model.add(GaussianNoise(0.01, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(LSTM(n_neuronas[0], activation=activation, name="encoder"))
        model.add(RepeatVector(train_X.shape[1]))
        model.add(LSTM(n_neuronas[1], activation=activation, name="decoder1"))
        model.add(RepeatVector(train_X.shape[1]))
        model.add(LSTM(n_neuronas[2], activation=activation, return_sequences=True, name="decoder2"))
        model.add(Flatten())
        model.add(Dense(n_neuronas[2], input_shape=(train_X.shape[1], train_X.shape[2]), activation="sigmoid"))
        model.add(Dense(1, activation="linear"))
        model.compile(loss='mae', optimizer=optimizer)
        model.summary()
        model = train_basic.train(model, train_X, train_y, test_X, test_y, n_neuronas[0], EPOCHS, activation, optimizer, cbs)
    
        
        return {"model":model, "train_X":train_X, "train_y":train_y, "test_X":test_X, 
                "test_y":test_y, "output":dict_ts["output"]}

    
    @staticmethod
    def train_lstm_autoencoder_avanzado2(dict_ts, n_neuronas, EPOCHS,activation='relu', optimizer="adam", callback=False):
        keras.backend.clear_session()
        train_X, train_y, test_X, test_y = dict_ts["trainX"], dict_ts["trainY"], dict_ts["testX"], dict_ts["testY"]
        
        
        model = Sequential()
        model.add(LSTM(64, batch_input_shape=(train_basic.BATCH_SIZE, train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(train_X, train_y, nb_epoch = EPOCHS, batch_size = train_basic.BATCH_SIZE)
        
        
    