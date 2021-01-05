# MPT, The Efficient Frontier, and LSTM's
## An open source project by Jacob Somer
### Goals of this project
1. Get historical stock market data programatically through an API
2. Forecast daily stock prices and covariance using Long Short Term Memory Networks
3. Visualize the [Efficient Frontier](https://en.wikipedia.org/wiki/Efficient_frontier) and discover new insights in the art of Stock market prediction

*note: if you only want to see step 3, please scroll down to the bottom*

### Helpful resources
For those who are just starting their learning journey in computational finance, here are some learning resources:

* [Youtube Video on Portfolio Management by MIT](https://www.youtube.com/watch?v=8TJQhQ2GZ0Y) 
* [Youtube Series on Machine Learning by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) 
* [Time Series Forecasting with Keras](https://keras.io/examples/timeseries/timeseries_weather_forecasting/) 
* [Modern Portfolio Theory Wiki](https://en.wikipedia.org/wiki/Modern_portfolio_theory) 


## The Dataset

This project uses AlphaVantage's [API](https://www.alphavantage.co/documentation/) to get historical data. The benefits of using Alpha Vantage is that their API is free to use. The downside is that they only allow 5 API calls per minute. To account for this, I parralelized the getData() function using the multiparallel library (lines 182-186 in Forecast.py):
```
num_cores = multiprocessing.cpu_count()
tmp = tqdm(stockList[i*5:i*5+5])

#parrallizing getting data because its faster
stockData = Parallel(n_jobs=num_cores)(delayed(getData)(j) for j in tmp)
``` 
Get data returns a list of historical data for any stock symbol you give it. In our case, get data returns the open and close price which we will later use as inputs for our returns and covariance prediction. Here is an example dataset plotted with Matplotlib:

![](NVDA_Daily.png)

A first order observation of this time series (which applies to most assets on the NYSE) is that there is an exponential growth pattern to the price. I accounted this by log transforming the data before feeding it to the LSTM (line 193 in Forecast.py):
```
2**predictNextDay(list(map(lambda x: np.log2(x), np.array(j[0])))
```

`np.array(j[0])` is a numpy array of historical prices. predictNextDay() is a function that trains our neural net and outputs a prediction for the next day. Inside the function, we use the sklearn MinMaxScaler() to scale our data from 0 to 1 which is necessary as our LSTM's reccurent activation layer uses a sigmoid function (lines 55-57 in Forecast.py):
```
scaler = MinMaxScaler()
scaler = scaler.fit((np.array(data)).reshape(-1, 1))
data_scaled = scaler.transform((np.array(data)).reshape(-1, 1))
```

## The Long Short-Term Memory Network

The reason for using an [LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) is because it has a fairly good track record when it comes to time-series forecasting. Unlike a traditional [Reccurent Neural Network](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), the lstm maintains both a cell state and a hidden state to pass contextual information. Basically, an [LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) is an [RNN](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) with gates. Here is an image to demonstrate the input, forget, output, and tanh layers respectively. Each of the first four functions represent a neural network. The last two functions represent the output cell state and hidden state to be passed on to the next [LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) layer. 

![](Structure-of-the-LSTM-cell-and-equations-that-describe-the-gates-of-an-LSTM-cell.jpg)

Luckily for us, Keras provides an easy to use API that does all of these operations given properly formatted data. We can even customize our activation functions with something like [RELU](https://www.google.com/search?q=relu&oq=relu&aqs=chrome.0.69i59j0i433i457j0i433l5j0.727j0j7&sourceid=chrome&ie=UTF-8) or [Sotftmax](https://en.wikipedia.org/wiki/Softmax_function), but for this project, I used [GPU compute](https://developer.nvidia.com/CUDNN) which requires our LSTM to have the feutures below: 
```
activation == tanh
recurrent_activation == sigmoid
recurrent_dropout == 0
unroll is False
use_bias is True
Inputs, if use masking, are strictly right-padded.
Eager execution is enabled in the outermost context.
```
In order to achieve better results given these requirements, I had to spend more time preprocessing (Log and MinMax scaling). This was important as it cut down overall execution time nearly 6 fold. It also used an average of 24% of my CPU rather than the 100% it was using without [GPU](https://developer.nvidia.com/CUDNN). Below I've posted the code for the entire predictNextDay() function to see the whole thing in action:
```

``` 


(https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) 
