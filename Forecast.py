#machine learning imports 

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout
from keras.layers import Dense


import keras
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.stats import pearsonr   

#data processing imports
import requests 
import pandas
from alpha_vantage.timeseries import TimeSeries
import time
from random import sample 

#parrallel computing imports
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


## given list of stocks 

def getCovariance(asset1,asset2):

    asset1Open=asset1[0]
    asset2Open=asset2[0]
    asset1Close=asset1[1]
    asset2Close=asset2[1]

    covarianceTimeSeries=[]
    for i in range(-min(len(asset1Close),len(asset2Close)),-1):
        list1=[]
        list2=[]
        for j in range(2):
            list1.append(asset1Open[i+j])
            list1.append(asset1Close[i+j])
            list2.append(asset2Open[i+j])
            list2.append(asset2Close[i+j])
        list1,list2=list(map(lambda x: np.log(1+x), np.array(list1))),list(map(lambda x: np.log(1+x), np.array(list2)))
        tmp=pearsonr(list1,list2)
        if not isinstance(tmp[0],np.float64):
            tmp=list(tmp)
            tmp[0]=0.001
        covarianceTimeSeries.append(float(tmp[0]))
    #using tanh beacuse it makes use of the cuDNN GPU making it much faster for
    # an operation that takes O(n^2) time
    return predictNextDay(covarianceTimeSeries, activationFunction="tanh")

def predictNextDay(data, activationFunction="tanh",lossFunction='mse',numberOfEpochs=10):
    scaler = MinMaxScaler()
    scaler = scaler.fit((np.array(data)).reshape(-1, 1))
    data_scaled = scaler.transform((np.array(data)).reshape(-1, 1))
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    # define input sequence
    raw_seq = data_scaled
    # choose a number of time steps
    n_steps = 50
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(128, activation= activationFunction, input_shape=(n_steps, n_features),return_sequences=False))
    # model.add(LSTM(128, activation=activationFunction, input_shape=(n_steps, n_features),return_sequences=False))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=lossFunction)
    # fit model
    history = model.fit(X, y, epochs=numberOfEpochs,batch_size=256,validation_split=0.1, verbose=1)
    x_input = data_scaled[-50:]
    x_input = x_input.reshape((1, n_steps, n_features))
    forecast = model.predict(x_input, verbose=0)
    forecast_copies = np.repeat(forecast[0][0], len(data), axis=-1).reshape(-1,1)
    y_pred_future = scaler.inverse_transform(forecast_copies)
    return y_pred_future[0][0]


def getData(stock):
    ts = TimeSeries(key='API Key Goes Here', output_format='pandas')
    try:
        currentDataFrame=ts.get_daily(symbol=stock,outputsize='full')[0]
    except ValueError:
        time.sleep(20)
        try:
            currentDataFrame=ts.get_daily(symbol=stock,outputsize='full')[0]
        except ValueError:
            time.sleep(20)
            try:
                currentDataFrame=ts.get_daily(symbol=stock,outputsize='full')[0]
            except ValueError:
                   return[list(range(1, 250))[::-1],list(range(1, 250))[::-1]]
    tmp=currentDataFrame['1. open'].astype(float).tolist()[:2000]
    tmp2=currentDataFrame['4. close'].astype(float).tolist()[:2000]
    tmp.reverse()
    tmp2.reverse()
    return [tmp,tmp2]

#helper methods
def Average(lst): 
    return sum(lst) / len(lst) 

def filterNegativeReturns(returns, data, stockList):
    newReturns=[]
    newData=[]
    newStockList=[]
    for i in range(len(returns)):
        if returns[i][2]<0:
            pass
        else:
            newReturns.append(returns[i])
            newData.append(data[i])
            newStockList.append(stockList[i])
    return newReturns,newData,newStockList 

def getSymbols(myList):
    df=pandas.read_csv("stockDF.csv")
    df = df.dropna(subset=['IPOyear'])
    df = df.dropna(subset=['industry'])
    df = df.dropna(subset=['MarketCap'])
    df = df.dropna(subset=['LastSale'])
    symbols=df["Symbol"].tolist()
    years=df["IPOyear"].tolist()
    samp=sample(symbols,40) 
    tmp=[]
    for i in range(len(samp)):
        if samp[i].isalpha() and years:
            tmp.append(samp[i])
    final=[]
    for j in myList:
        final.append(j)
    for k in tmp:
        final.append(k)
    while len(final)%5!=0:
        final.pop(-1)
    return final



if __name__ == "__main__":
    totalTime = time.time()
  
    inputs=['AMZN','BA','GOOGL',"TSLA","ALB","HSY","AAPL","AAL","PTON","GS","PFE",'BABA', 'BILI', 'PDD', 'GME', 'CRWD', 'NIO', 'NVAX', 'MRNA', 'RIOT', 'MARA', 'GPRO', 'ZM', 'NFLX', 'AMD', 'NCLH', 'LULU', 'ROKU', 'FB', 'PLTR', 'TGT', 'WMT', 'SPWR', 'APPN', 'PINS', 'FSLR', 'FSLY', 'INO', 'FCEL', 'FUBO']    #remove duplicates
    stockList=[]
    [stockList.append(x) for x in inputs if x not in stockList]
    # stockList=getSymbols(stockList)

    #list to store time of each neural network iteration
    times=[] 

    #list to store returns in order
    returns=[] 

    #list that contains all the covariance data
    data=[]


    if len(stockList)%5==0:
        remaining=len(stockList)
        for i in range(int(len(stockList)/5)):

            num_cores = multiprocessing.cpu_count()
            tmp = tqdm(stockList[i*5:i*5+5])

            #parrallizing getting data because its faster
            stockData = Parallel(n_jobs=num_cores)(delayed(getData)(j) for j in tmp)
        
            predictions=[]
            for j in stockData:
                start_time = time.time()
                #use relu for stock prediction because it deals better with exponential data
                # predictions.append(2**predictNextDay(list(map(lambda x: np.log2(x), np.array(j[0]))),activationFunction="tanh",lossFunction='mean_squared_logarithmic_error',numberOfEpochs=50))
                predictions.append(2**predictNextDay(list(map(lambda x: np.log2(x), np.array(j[0]))),activationFunction="tanh",lossFunction='mean_absolute_error',numberOfEpochs=150))
                times.append(time.time() - start_time)
                remaining-=1
                print("time remaining to calculate returns: %s ",(Average(times)*remaining)/60, " minutes")
    
            for j in range(len(predictions)):
                data.append(stockData[j])
                #get the percent returns and append it to the list
                original=float(stockData[j][1][-1])
                prediction=predictions[j]
                returns.append([original,prediction,((prediction-original)/original)])
        returns_df=pandas.DataFrame(np.array(returns),columns=["Original Price","Predicted Price","Predicted Returns"],index=stockList)
        returns_df.to_csv("returns.csv") 
        returns,data,stockList=filterNegativeReturns(returns,data,stockList)
        #DataFrame to store covariance predictions
        covarianceDataFrame= pandas.DataFrame(np.array([[None for x in range(len(stockList))] for x in range(len(stockList))]),columns=stockList,index=stockList)

        #given returns 
        #get covariance for each pair and add it to a dataFrame
        operations=(len(stockList)*(len(stockList)-1)/2)

    
        for k in range(len(stockList)-1):
            for j in range(k+1,len(stockList)):
                print("Predicting covariance for: ", stockList[k], ' and ', stockList[j])
                print("time remaining: %s ",(Average(times)*operations)/60, " minutes")
                operations-=1
                val=getCovariance(data[k],data[j])
                covarianceDataFrame.at[stockList[k],stockList[j]]=val
                covarianceDataFrame.at[stockList[j],stockList[k]]=val
                
            covarianceDataFrame.at[stockList[k],stockList[k]]=1
            if k==len(stockList)-1:
                covarianceDataFrame.at[stockList[k+1],stockList[k+1]]=1
                
       
        covarianceDataFrame.to_csv("covariance.csv") 
    

        print("--- %s seconds ---" % str(time.time() - totalTime))
    



      
