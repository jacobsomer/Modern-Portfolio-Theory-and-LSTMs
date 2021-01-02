import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data, meta_data = ts.get_daily(symbol='NVDA', outputsize='full')
data['4. close'].plot()
plt.title('Daily Times Series for the NVDA stock')
plt.savefig('MFST_Daily.png')
plt.show()

if __name__ == "__main__": 
    preds=pd.read_csv("returns.csv")
    preds=preds.rename(columns={"Unnamed: 0": "Symbols"})
    preds=preds.set_index('Symbols')
    preds=preds.drop(['FUBO','MRNA','MARA','ZM','PINS','GME','NVAX'], axis=0)
    predReturns=preds['Predicted Returns'].astype(float).tolist()
    covariance=pd.read_csv("covariance.csv")
    covariance=covariance.rename(columns={"Unnamed: 0": "Symbols"})
    covariance=covariance.set_index('Symbols')
    covariance=covariance.drop(['FUBO','MRNA','MARA','ZM','GME','NVAX'], axis=1)
    covariance=covariance.drop(['FUBO','MRNA','MARA','ZM','GME','NVAX'], axis=0)
    stockList=[]
    for col in covariance.columns: 
        stockList.append(col)

    variance=[]
    returns=[]
    portfolios=[]
    for i in range(50000):
        samples = np.random.randint(0, 10,  len(stockList)) 
        total=sum(list(samples))
        normalised = samples/total
        normalised=list(normalised)
        myDict={}
        for j in range(len(stockList)):
            myDict[stockList[j]]=normalised[j]

        var = covariance.mul(myDict, axis=0).mul(myDict, axis=1).sum().sum()

        r=0
        for d in range(len(normalised)):
            r+=normalised[d]*predReturns[d]
        returns.append(r*100)
        variance.append(var)
        portfolios.append(normalised)
    
    
    best=[30,0,0]
    for t in range(len(variance)):
        if returns[t]>1.1 and best[0]>variance[t]:
            best[0]=variance[t]
            best[1]=returns[t]
            best[2]=portfolios[t]
    amount={}
    for i in range(len(best[2])):
        amount[stockList[i]]=(best[2][i]*100)
    print(amount)
    print("Mean Variance: ",best[0])
    print("Expected Return: ",best[1])

    sharpeRatio=[]

    for k in range(len(returns)):
        ratio=(returns[k]-.00233)/variance[k]
        sharpeRatio.append(ratio)


    frontierx=[]
    frontiery=[]
    def isOptimal(r,v,returns1,volatility):
        for i in range(len(returns1)):
            if returns1[i]>r and volatility[i]<v:
                return False
        return True
    for j in range(len(returns)):
        # if len(frontierx)>10:
        #   break
        if isOptimal(returns[j],variance[j],returns,variance):
            frontierx.append(variance[j])
            frontiery.append(returns[j])


    power = np.array([x for _,x in sorted(zip(frontierx,frontiery))])
    T = np.array(sorted(frontierx))



    xsmoothed = gaussian_filter1d(T, sigma=2)
    ysmoothed = gaussian_filter1d(power, sigma=2)

    fig, ax = plt.subplots(figsize=(20, 10),nrows=1, ncols=1)
    ax.set_facecolor(((240/255),240/255,240/255))
    plt.grid(True, linewidth=0.5, color='#3B8E7C', linestyle='-')
    fig.patch.set_facecolor((240/255,240/255,240/255))
    plt.scatter(variance, returns, c=sharpeRatio, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.plot(xsmoothed, ysmoothed, 'r-')
    plt.savefig('cover.png')
    plt.show()

   