
# S&P 500 Stock Clustering

This notebook demonstrates a clustering of the S&P 500 stock exchange, based on a select set of financial figures

The exchange consists of 500 companies, but includes 505 common stocks, due to 5 companies having two shares of stocks in the exchange (Facebook, Under-Armour, NewsCorp, Comcast and 21st Century Fox)


```python
# libraries for making requests and parsing HTML
import requests
from bs4 import BeautifulSoup

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn for kmeans and model metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# pandas, for data wrangling
import pandas as pd
```

    /anaconda3/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)


# Data Accquisition

For the data I wanted access to, the existing APIs for financial data did not work out. Instead. I decided to manually scrape the data, ussing Wikipedia and Yahoo Finance.

1. scrape the list of S&P 500 tickers from Wikipedia
2. scrape the financial figures for each stock ticker from Yahoo Finance


```python
# URL to get S&P tickers from
TICKER_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# multi-level identifier, to select each row of ticker table in HTML response
TABLE_IDENTIFIER = '#constituents tbody tr td'

# yahoo finance URL we can use to scrape data for each company
YAHOO_URL = 'http://finance.yahoo.com/quote/'

# HTML classes for various elements on yahoo finance page

YAHOO_TABLE_CLASS = 'Ta(end) Fw(600) Lh(14px)'
# EPS (TTM) react-id
# Open price react-id
# Div/Yield react-id
YAHOO_IDS = ['OPEN-value', 'EPS_RATIO-value', 'DIVIDEND_AND_YIELD-value', 'PE_RATIO-value']
```


```python
# get HTML content from wikipedia S&P 500 page
res = BeautifulSoup(requests.get(TICKER_URL).text, 'html.parser')
```


```python
# get the table of stock ticker data, selecting on TABLE_ID
table_data = [ticker for ticker in res.select(TABLE_IDENTIFIER)]
```


```python
# iterate over each row of table (9 elements of information), and extract the individual tickers
tickers = [table_data[i].text for i in range(0, len(table_data), 9)]
```


```python
# iterate through the S&P 500 company tickers, and collect data from Yahoo Finance
def get_yahoo_ticker_data(tickers):
    ticker_data = []
    # make GET request for specified ticker
    print(len(tickers))
    for i, ticker in enumerate(tickers):
        print(i)
        try:
            REQ_URL = YAHOO_URL + ticker[:-1] + '?p=' + ticker[:-1]
            ticker_i_res = requests.get(REQ_URL)
            ticker_i_parser = BeautifulSoup(ticker_i_res.text, 'html.parser')

            ticker_i_data = [ticker[:-1]]
            ticker_i_open_eps_div = [ticker_i_parser.find(attrs={'class': YAHOO_TABLE_CLASS, 'data-test': id_}).text for id_ in YAHOO_IDS]
            for data in ticker_i_open_eps_div:
                    ticker_i_data.append(data)
            ticker_data.append(ticker_i_data)
        except:
            print("error for " + ticker)
            continue
    return ticker_data
```

# Saving the data

The process of scraping all of the necessary data was rather cumbersome, so it made sense to save the data to file for future experiments


```python
# convert yahoo finance data to dataframe

# will include:
# EPS (TTM) => earnings per share for trailing 12 months
# Dividend/Yield => dividend per share / price per share
# P/E ratio => share price / earnings per share
try:
    df = pd.read_csv('data.csv')
except:
    # iterate over stock tickers, and get 1 year of time-series data
    market_data = pd.DataFrame()
    yahoo_data = get_yahoo_ticker_data(tickers)
    df = pd.DataFrame(yahoo_data, columns=['ticker', 'open', 'eps', 'div'])#, 'pe'],)
    df.to_csv(path_or_buf='data.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ticker</th>
      <th>open</th>
      <th>eps</th>
      <th>div</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>MMM</td>
      <td>169.78</td>
      <td>8.43</td>
      <td>5.76 (3.39%)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>ABT</td>
      <td>87.08</td>
      <td>1.84</td>
      <td>1.44 (1.65%)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>ABBV</td>
      <td>90.05</td>
      <td>2.18</td>
      <td>4.72 (5.24%)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>ABMD</td>
      <td>179.85</td>
      <td>4.79</td>
      <td>N/A (N/A)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>ACN</td>
      <td>203.60</td>
      <td>7.36</td>
      <td>3.72 (1.83%)</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['div'] = df['div'].replace({'N/A (N/A)': 0})
```

# Preprocessing

Some data preprocessing is required before proceeding forward with experimentation

1. separating percentage dividend yield and dividend yield amount into two separate featuress
2. reformatting some features into representations that could be converted to numerical types
3. casting features of DataFrame to numerical types


```python
# drop NaN values
df = df.dropna()

# remove NaN values that aren't using NaN value
#df = df[df['eps'] != 'N/A']
df['eps'] = df['eps'].astype(float)


# preprocess open values
df['open'] = df['open'].astype(str)
df['open'] = df['open'].apply(lambda x: x.replace(',', '')).astype(float)

# split dividend into amount and percentage
df['div'] = df['div'].astype(str)
df['div_pct'] = df['div'].apply(lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 else '(0%)')
df['div_pct'] = df['div_pct'].apply(lambda x: x[1:-2]).astype(float)
df['div_amt'] = df['div'].apply(lambda x: x.split(' ')[0]).astype(float)
df = df.drop(['div'], axis=1)
df.isnull().sum()
```




    Unnamed: 0    0
    ticker        0
    open          0
    eps           0
    div_pct       0
    div_amt       0
    dtype: int64




```python
# relevant data for now, will be using these columns for k-means clustering
two_dim_cluster_data = df[['ticker', 'eps', 'div_pct']]
four_dim_cluster_data = df[['ticker', 'eps', 'open', 'div_pct', 'div_amt']]
```


```python
sns.scatterplot(x='eps', y='div_pct', data=two_dim_cluster_data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a10f6b6a0>




![png](output_14_1.png)


# Clustering the data: The K-Means algorithm

Now that the data the accquisition and preprocessing was complete, the next step is clustering our stock data, analyzing the performance of the clustering, based on the number of centroids, and then generating a final clustering based on some performance metrics.

The K-means algorithm operates as follows:

    1. a number of "centroids" are randomly initialized (the number of hyperparameter of the model), these centroid
       match the dimension of the feature set, and can be imagine as a vector into some n-dimensional space
    2. every sample in the data set is then compared to each of the randomly initialized centroids, to see how far 
       it is away from the centroid. Since the samples and centroids are vectors, the distance 
       between a vector v and a centroid u is the vector normal of the difference between the two vectors 
       ((u1-v1)^2 + (u2-v2)^2 + ....)^(1/2). Each sample is then "clustered" with the centroid it is closest to.
    3. After each sample has been clustered with a specific centroid, each centroid is repositioned, such that it
       is the average of all of the samples that have been clustered with it.
    4. The sample association and centroid repositioning steps are then repeated for some number of iterations


```python
# iterate over a variety of amounts of cluster centroids for clustering our stock data
# looking for an "elbow" in the sum of squared error plot, for different amounts of centroids
def k_means_func(data, max_centroids=25):
    # transform numerical features (eps and percentage dividend)
    transform_data = StandardScaler().fit_transform(data.iloc[:,1:])
    
    sum_square_err = {}
    sil_score = {}
    for num_centroids in range(2,max_centroids):
        model = KMeans(n_clusters=num_centroids, random_state=2, n_init=10)
        model.fit(transform_data)
        sum_square_err[num_centroids] = model.inertia_
        sil_score[num_centroids] = silhouette_score(transform_data, model.labels_, random_state=2)
    
    plt.figure(figsize=(16,6))
    ax1 = plt.subplot(211)
    plt.plot(list(sum_square_err.keys()), list(sum_square_err.values()))
    ax1.title.set_text("k-means sum squared error")
    plt.xlabel("num. centroids")
    plt.ylabel("sum squared error")
    plt.xticks([i for i in range(2, max_centroids)])
    
    ax2 = plt.subplot(212)
    plt.plot(list(sil_score.keys()), list(sil_score.values()))
    ax2.title.set_text("k-means silhouette score")
    plt.xlabel("num. centroids")
    plt.ylabel("score")
    plt.xticks([i for i in range(2, max_centroids)])
    plt.yticks([i / 10 for i in range(10)])
```

# Measuring the performance of K-Means clustering

The K-means algorithm cannot be measured in performance in the same way as supervised learning algorithms. There is no prediction error, since the data we are given is unlabeled, and instead, we measure the performance of the k-means algorithm based on the ability of the chosen number of centroids to effectively cluster the data. Notely, one of the common metrics for K-means is measuring the squared sum of errors between each sample and the centroid it is clustered with, where the squared error is just the squared vector normal of the difference between the sample and the centroid

In addition to the squared sum of errors, K-means is often measured using the silhouette score. This metric is the mean of the silhouette coefficient for every sample. The silhouette coefficient can be defined as follows:

* for a sample S, we define A(S) as the mean distance between S and every other element in S's assigned cluster
* we define B(S) as the mean distance between S, and every point in the closest cluster to S, other than S's assigned cluster
* we define SC(S), the silhouette coefficient, as the difference between A(S) and B(S), divided by the larger of A(S) and B(S)
* therefore, SC(S) ranges from 0 to 1, where SC(S) = 1 means the mean distance from S to every point in S's cluster is 0, and SC(S) = 0 means that the mean distance from S to every point in its cluster is the same as the mean distance from S to every point in the nearest other cluster

Below, we plot these metrics for our application of K-means to the stock data, we can see the following:

1. The silhouette score drops rather quickly after n grows greater than 3-4, this implies that a small amount of clusters most likely results in a few disparate clusters (with a single cluster comprising much of the data)
2. The silhouette score stabilizes after it drops to ~0.4, while the SSE continues to drop rapidly until n~10
3. The silhouette score bumps up slightly for a few values of n (n = 11, n = 15, n = 20), these are likely good values 
   for n, since the silhouette score is stable but slightly up, while the SSE continues to go down 


```python
k_means_func(two_dim_cluster_data)
```


![png](output_18_0.png)



```python
k_means_func(four_dim_cluster_data)    
```


![png](output_19_0.png)


# Finalizing our clusterings

Given that we have identified a few values for our centroid hyperparameter that seem fruitful, the next step is to fit and cluster the data for these specified values, our results will not be predictions of an output variable, as
is the case in supervised learning, but rather, predictions of certain groupings of our stock tickers


```python
def classify_four_dim_stocks(data, cluster_configs):
    transform_data = StandardScaler().fit_transform(data.iloc[:,1:])
    # initialize K-means models with each of the specified cluster hyperparameter valuess
    for config in cluster_configs.keys():
        model = KMeans(n_clusters=cluster_configs[config], random_state=5, n_init=10)
        model.fit(transform_data)
        data[config] = model.labels_
    return data
```


```python
cluster_config_one = {
    'cluster_five': 5,
    'cluster_ten': 10,
    'cluster_fourteen': 14,
    'cluster_twenty': 20
}
four_dim_cluster_data = classify_four_dim_stocks(four_dim_cluster_data[['ticker', 'eps', 'open', 'div_pct', 'div_amt']], cluster_config_one)
```

    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      import sys



```python
four_dim_cluster_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>eps</th>
      <th>open</th>
      <th>div_pct</th>
      <th>div_amt</th>
      <th>cluster_five</th>
      <th>cluster_ten</th>
      <th>cluster_fourteen</th>
      <th>cluster_twenty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>8.43</td>
      <td>169.78</td>
      <td>3.39</td>
      <td>5.76</td>
      <td>0</td>
      <td>4</td>
      <td>11</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ABT</td>
      <td>1.84</td>
      <td>87.08</td>
      <td>1.65</td>
      <td>1.44</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABBV</td>
      <td>2.18</td>
      <td>90.05</td>
      <td>5.24</td>
      <td>4.72</td>
      <td>0</td>
      <td>4</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABMD</td>
      <td>4.79</td>
      <td>179.85</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>7.36</td>
      <td>203.60</td>
      <td>1.83</td>
      <td>3.72</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ATVI</td>
      <td>2.11</td>
      <td>58.34</td>
      <td>0.63</td>
      <td>0.37</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ADBE</td>
      <td>6.00</td>
      <td>322.10</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>16</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AMD</td>
      <td>0.19</td>
      <td>42.79</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AAP</td>
      <td>6.17</td>
      <td>158.13</td>
      <td>0.16</td>
      <td>0.24</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AES</td>
      <td>0.76</td>
      <td>18.88</td>
      <td>3.03</td>
      <td>0.57</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AMG</td>
      <td>-3.35</td>
      <td>86.68</td>
      <td>1.50</td>
      <td>1.28</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>AFL</td>
      <td>4.05</td>
      <td>53.33</td>
      <td>2.03</td>
      <td>1.08</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>A</td>
      <td>3.37</td>
      <td>83.75</td>
      <td>0.85</td>
      <td>0.72</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>APD</td>
      <td>7.94</td>
      <td>235.09</td>
      <td>1.98</td>
      <td>4.64</td>
      <td>0</td>
      <td>4</td>
      <td>11</td>
      <td>13</td>
    </tr>
    <tr>
      <th>14</th>
      <td>AKAM</td>
      <td>2.74</td>
      <td>84.44</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ALK</td>
      <td>4.92</td>
      <td>70.41</td>
      <td>2.03</td>
      <td>1.40</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ALB</td>
      <td>5.38</td>
      <td>68.90</td>
      <td>2.22</td>
      <td>1.47</td>
      <td>2</td>
      <td>5</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ARE</td>
      <td>1.09</td>
      <td>155.29</td>
      <td>2.63</td>
      <td>4.12</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>13</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ALXN</td>
      <td>6.52</td>
      <td>109.43</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ALGN</td>
      <td>5.21</td>
      <td>269.48</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ALLE</td>
      <td>4.79</td>
      <td>123.53</td>
      <td>0.87</td>
      <td>1.08</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>AGN</td>
      <td>-27.98</td>
      <td>190.50</td>
      <td>1.56</td>
      <td>2.96</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ADS</td>
      <td>8.81</td>
      <td>109.78</td>
      <td>2.31</td>
      <td>2.52</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LNT</td>
      <td>2.24</td>
      <td>54.08</td>
      <td>2.64</td>
      <td>1.42</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>24</th>
      <td>ALL</td>
      <td>7.32</td>
      <td>110.18</td>
      <td>1.82</td>
      <td>2.00</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>25</th>
      <td>GOOGL</td>
      <td>46.60</td>
      <td>1357.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>GOOG</td>
      <td>46.60</td>
      <td>1356.60</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>MO</td>
      <td>0.93</td>
      <td>50.90</td>
      <td>6.61</td>
      <td>3.36</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>28</th>
      <td>AMZN</td>
      <td>22.57</td>
      <td>1795.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>18</td>
    </tr>
    <tr>
      <th>29</th>
      <td>AMCR</td>
      <td>0.31</td>
      <td>10.75</td>
      <td>4.26</td>
      <td>0.46</td>
      <td>3</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>474</th>
      <td>V</td>
      <td>5.32</td>
      <td>185.52</td>
      <td>0.65</td>
      <td>1.20</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>15</td>
    </tr>
    <tr>
      <th>475</th>
      <td>VNO</td>
      <td>15.73</td>
      <td>65.26</td>
      <td>4.06</td>
      <td>2.64</td>
      <td>3</td>
      <td>0</td>
      <td>13</td>
      <td>17</td>
    </tr>
    <tr>
      <th>476</th>
      <td>VMC</td>
      <td>4.50</td>
      <td>143.12</td>
      <td>0.87</td>
      <td>1.24</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>477</th>
      <td>WRB</td>
      <td>3.61</td>
      <td>69.64</td>
      <td>0.63</td>
      <td>0.44</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>478</th>
      <td>WAB</td>
      <td>1.46</td>
      <td>74.31</td>
      <td>0.64</td>
      <td>0.48</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>479</th>
      <td>WMT</td>
      <td>5.00</td>
      <td>121.51</td>
      <td>1.75</td>
      <td>2.12</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>480</th>
      <td>WBA</td>
      <td>4.31</td>
      <td>57.23</td>
      <td>3.21</td>
      <td>1.83</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>481</th>
      <td>DIS</td>
      <td>6.64</td>
      <td>147.77</td>
      <td>1.19</td>
      <td>1.76</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>15</td>
    </tr>
    <tr>
      <th>482</th>
      <td>WM</td>
      <td>4.09</td>
      <td>113.02</td>
      <td>1.81</td>
      <td>2.05</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>483</th>
      <td>WAT</td>
      <td>8.13</td>
      <td>231.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>484</th>
      <td>WEC</td>
      <td>3.45</td>
      <td>91.33</td>
      <td>2.78</td>
      <td>2.53</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>485</th>
      <td>WCG</td>
      <td>12.44</td>
      <td>317.40</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>16</td>
    </tr>
    <tr>
      <th>486</th>
      <td>WFC</td>
      <td>4.65</td>
      <td>54.46</td>
      <td>3.75</td>
      <td>2.04</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>17</td>
    </tr>
    <tr>
      <th>487</th>
      <td>WELL</td>
      <td>2.80</td>
      <td>76.97</td>
      <td>4.50</td>
      <td>3.48</td>
      <td>3</td>
      <td>4</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>488</th>
      <td>WDC</td>
      <td>-5.26</td>
      <td>57.17</td>
      <td>3.49</td>
      <td>2.00</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>17</td>
    </tr>
    <tr>
      <th>489</th>
      <td>WU</td>
      <td>2.60</td>
      <td>26.87</td>
      <td>2.98</td>
      <td>0.80</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>490</th>
      <td>WRK</td>
      <td>3.33</td>
      <td>41.87</td>
      <td>4.43</td>
      <td>1.86</td>
      <td>3</td>
      <td>0</td>
      <td>10</td>
      <td>17</td>
    </tr>
    <tr>
      <th>491</th>
      <td>WY</td>
      <td>-0.21</td>
      <td>29.74</td>
      <td>4.59</td>
      <td>1.36</td>
      <td>3</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>492</th>
      <td>WHR</td>
      <td>16.58</td>
      <td>147.25</td>
      <td>3.27</td>
      <td>4.80</td>
      <td>0</td>
      <td>4</td>
      <td>11</td>
      <td>13</td>
    </tr>
    <tr>
      <th>493</th>
      <td>WMB</td>
      <td>0.12</td>
      <td>23.04</td>
      <td>6.63</td>
      <td>1.52</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>494</th>
      <td>WLTW</td>
      <td>6.74</td>
      <td>201.02</td>
      <td>1.29</td>
      <td>2.60</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>15</td>
    </tr>
    <tr>
      <th>495</th>
      <td>WYNN</td>
      <td>6.16</td>
      <td>138.00</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>496</th>
      <td>XEL</td>
      <td>2.50</td>
      <td>63.57</td>
      <td>2.55</td>
      <td>1.62</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>497</th>
      <td>XRX</td>
      <td>2.84</td>
      <td>36.93</td>
      <td>2.69</td>
      <td>1.00</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>498</th>
      <td>XLNX</td>
      <td>3.71</td>
      <td>96.26</td>
      <td>1.54</td>
      <td>1.48</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>499</th>
      <td>XYL</td>
      <td>2.80</td>
      <td>78.10</td>
      <td>1.23</td>
      <td>0.96</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>500</th>
      <td>YUM</td>
      <td>3.62</td>
      <td>99.48</td>
      <td>1.69</td>
      <td>1.68</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZBH</td>
      <td>-0.44</td>
      <td>149.90</td>
      <td>0.64</td>
      <td>0.96</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZION</td>
      <td>4.27</td>
      <td>51.60</td>
      <td>2.64</td>
      <td>1.36</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>503</th>
      <td>ZTS</td>
      <td>3.02</td>
      <td>127.15</td>
      <td>0.63</td>
      <td>0.80</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>497 rows × 9 columns</p>
</div>




```python
def output_cluster_tickers(original_data, cluster_data, cluster, show_tickers=[]): 
    for i in range(0, max(cluster_data[cluster])):
        if(i in show_tickers or len(show_tickers) == 0):
            # list of tickers for the current cluster
            ticker_list = list(cluster_data[cluster_data[cluster] == i]['ticker'])
            print("cluster " + str(i) + ":")
            print("includes " + str(len(ticker_list)) + " stocks")
            print(ticker_list)
            # original data for tickers that are part of cluster, more useful than
            # the transformed data
            curr_data = original_data[original_data['ticker'].isin(ticker_list)]
            print(curr_data[['open', 'div_pct', 'div_amt', 'eps']].mean())
            print()
```


```python
output_cluster_tickers(df, four_dim_cluster_data, 'cluster_twenty')
```

    cluster 0:
    includes 24 stocks
    ['AMCR', 'APA', 'T', 'CAH', 'CNP', 'COTY', 'F', 'BEN', 'GPS', 'HRB', 'HBI', 'HST', 'HBAN', 'IPG', 'KIM', 'KMI', 'KHC', 'NWL', 'NLSN', 'PBCT', 'PPL', 'SLB', 'TPR', 'WY']
    open       23.628750
    div_pct     4.732500
    div_amt     1.106667
    eps        -0.760000
    dtype: float64
    
    cluster 1:
    includes 27 stocks
    ['ABBV', 'BXP', 'CVX', 'CCI', 'DRI', 'DLR', 'D', 'DTE', 'DUK', 'ETR', 'EXR', 'XOM', 'FRT', 'SJM', 'KMB', 'LYB', 'MAA', 'OKE', 'PM', 'PSX', 'PNW', 'PRU', 'SLG', 'UPS', 'VLO', 'WELL', 'WYNN']
    open       105.968148
    div_pct      3.819630
    div_amt      3.918148
    eps          4.584444
    dtype: float64
    
    cluster 2:
    includes 82 stocks
    ['ABT', 'AMG', 'AFL', 'A', 'ALK', 'ALLE', 'AAL', 'APH', 'AOS', 'AMAT', 'APTV', 'BLL', 'BAC', 'BAX', 'BWA', 'CBOE', 'CERN', 'SCHW', 'CHD', 'XEC', 'CTXS', 'CTSH', 'CMCSA', 'CTVA', 'CSX', 'DHI', 'DVN', 'FANG', 'DD', 'ETFC', 'EBAY', 'EOG', 'EFX', 'EXPE', 'EXPD', 'FIS', 'FRC', 'FLIR', 'FLS', 'FMC', 'FBHS', 'FOXA', 'FOX', 'FCX', 'GL', 'HIG', 'HES', 'HRL', 'ICE', 'JBHT', 'LW', 'LDOS', 'MRO', 'MAS', 'MCK', 'MGM', 'MCHP', 'MOS', 'NEM', 'NWSA', 'NWS', 'NKE', 'NBL', 'ORCL', 'PCAR', 'PNR', 'PRGO', 'PHM', 'RJF', 'RHI', 'ROL', 'ROST', 'SEE', 'LUV', 'TJX', 'TSCO', 'VRSK', 'VMC', 'XLNX', 'XYL', 'ZBH', 'ZTS']
    open       71.278902
    div_pct     1.359146
    div_amt     0.892927
    eps         2.791463
    dtype: float64
    
    cluster 3:
    includes 1 stocks
    ['NVR']
    open       3820.00
    div_pct       0.00
    div_amt       0.00
    eps         215.31
    dtype: float64
    
    cluster 4:
    includes 3 stocks
    ['GOOGL', 'GOOG', 'AZO']
    open       1311.956667
    div_pct       0.000000
    div_amt       0.000000
    eps          52.210000
    dtype: float64
    
    cluster 5:
    includes 66 stocks
    ['AGN', 'ADS', 'ALL', 'AEE', 'AEP', 'AWK', 'ABC', 'ADI', 'AJG', 'AIZ', 'ATO', 'AVY', 'BBY', 'BR', 'CHRW', 'CE', 'CB', 'CINF', 'C', 'STZ', 'CVS', 'DFS', 'DOV', 'ETN', 'EMR', 'EQR', 'ES', 'FDX', 'GRMN', 'GPC', 'HAS', 'HSY', 'IR', 'IFF', 'LLY', 'LOW', 'MMC', 'MDT', 'MRK', 'MSI', 'NDAQ', 'NTRS', 'PAYX', 'PPG', 'PG', 'PLD', 'QCOM', 'DGX', 'RL', 'RSG', 'SWKS', 'SWK', 'SBUX', 'STT', 'SYY', 'TROW', 'TGT', 'TEL', 'TIF', 'TSN', 'UTX', 'VFC', 'WMT', 'WM', 'WEC', 'YUM']
    open       110.851970
    div_pct      2.137273
    div_amt      2.301212
    eps          4.087879
    dtype: float64
    
    cluster 6:
    includes 3 stocks
    ['IBM', 'PSA', 'SPG']
    open       161.196667
    div_pct      4.833333
    div_amt      7.626667
    eps          8.170000
    dtype: float64
    
    cluster 7:
    includes 64 stocks
    ['ATVI', 'AMD', 'AKAM', 'ALXN', 'AME', 'ARNC', 'ADSK', 'BSX', 'CDNS', 'CPRI', 'KMX', 'CBRE', 'CNC', 'CXO', 'CPRT', 'DHR', 'DVA', 'XRAY', 'DISCA', 'DISCK', 'DISH', 'DLTR', 'FISV', 'FTNT', 'FTV', 'IT', 'GE', 'GPN', 'HSIC', 'HLT', 'HOLX', 'INFO', 'INCY', 'IPGP', 'IQV', 'JEC', 'KEYS', 'LEN', 'LKQ', 'L', 'MU', 'MNST', 'MYL', 'NOV', 'NCLH', 'NRG', 'PYPL', 'PKI', 'PGR', 'PVH', 'QRVO', 'PWR', 'CRM', 'SNPS', 'TMUS', 'TTWO', 'TXT', 'TRIP', 'TWTR', 'UAA', 'UA', 'VAR', 'WRB', 'WAB']
    open       78.296094
    div_pct     0.152500
    div_amt     0.110156
    eps         2.325156
    dtype: float64
    
    cluster 8:
    includes 4 stocks
    ['BIIB', 'MHK', 'REGN', 'SIVB']
    open       264.5475
    div_pct      0.0000
    div_amt      0.0000
    eps         29.5425
    dtype: float64
    
    cluster 9:
    includes 10 stocks
    ['MO', 'CTL', 'HP', 'IVZ', 'IRM', 'LB', 'MAC', 'M', 'OXY', 'WMB']
    open       27.793
    div_pct     7.789
    div_amt     2.130
    eps         0.224
    dtype: float64
    
    cluster 10:
    includes 56 stocks
    ['AES', 'ALB', 'LNT', 'AIG', 'AIV', 'ADM', 'BK', 'BMY', 'COG', 'CPB', 'CF', 'CSCO', 'CFG', 'CMS', 'KO', 'CL', 'CAG', 'COP', 'GLW', 'DAL', 'DRE', 'DXC', 'EVRG', 'EXC', 'FAST', 'FITB', 'FE', 'HAL', 'HPE', 'HFC', 'HPQ', 'INTC', 'JCI', 'JNPR', 'KEY', 'KR', 'LEG', 'LNC', 'MXIM', 'MDLZ', 'MS', 'NTAP', 'NI', 'NUE', 'PEG', 'RF', 'SYF', 'FTI', 'UDR', 'USB', 'UNM', 'WBA', 'WU', 'XEL', 'XRX', 'ZION']
    open       44.340000
    div_pct     2.839643
    div_amt     1.240000
    eps         2.648036
    dtype: float64
    
    cluster 11:
    includes 4 stocks
    ['BLK', 'AVGO', 'EQIX', 'LMT']
    open       443.6425
    div_pct      2.7200
    div_amt     11.4100
    eps         14.8175
    dtype: float64
    
    cluster 12:
    includes 1 stocks
    ['BKNG']
    open       2008.67
    div_pct       0.00
    div_amt       0.00
    eps          97.36
    dtype: float64
    
    cluster 13:
    includes 35 stocks
    ['ACN', 'APD', 'ARE', 'AMT', 'AMP', 'ADP', 'CAT', 'CLX', 'DE', 'HON', 'HII', 'ITW', 'JNJ', 'JPM', 'KLAC', 'LRCX', 'LIN', 'MTB', 'MCD', 'NEE', 'NSC', 'PKG', 'PH', 'PEP', 'PNC', 'RTN', 'ROK', 'RCL', 'SRE', 'SNA', 'TXN', 'TRV', 'UNP', 'UNH', 'WHR']
    open       181.172857
    div_pct      2.270571
    div_amt      3.953714
    eps          9.180000
    dtype: float64
    
    cluster 14:
    includes 5 stocks
    ['CMG', 'ISRG', 'MTD', 'ORLY', 'TDG']
    open       642.804
    div_pct      0.000
    div_amt      0.000
    eps         14.996
    dtype: float64
    
    cluster 15:
    includes 38 stocks
    ['AXP', 'ANTM', 'AON', 'AAPL', 'BDX', 'COF', 'CDW', 'CTAS', 'CME', 'COST', 'DG', 'ECL', 'EL', 'HCA', 'HUM', 'IEX', 'INTU', 'JKHY', 'KSU', 'LHX', 'MKTX', 'MAR', 'MLM', 'MA', 'MKC', 'MSFT', 'MCO', 'MSCI', 'PXD', 'RMD', 'ROP', 'SPGI', 'SBAC', 'SYK', 'TFX', 'V', 'DIS', 'WLTW']
    open       219.314474
    div_pct      1.013947
    div_amt      2.072105
    eps          7.210789
    dtype: float64
    
    cluster 16:
    includes 30 stocks
    ['ABMD', 'ADBE', 'AAP', 'ALGN', 'ANSS', 'ANET', 'CHTR', 'CI', 'COO', 'EW', 'EA', 'FFIV', 'FB', 'FLT', 'IDXX', 'ILMN', 'LH', 'NFLX', 'NVDA', 'ODFL', 'NOW', 'TMO', 'ULTA', 'UAL', 'URI', 'UHS', 'VRSN', 'VRTX', 'WAT', 'WCG']
    open       235.018000
    div_pct      0.055667
    div_amt      0.107333
    eps          7.425000
    dtype: float64
    
    cluster 17:
    includes 31 stocks
    ['CCL', 'CMA', 'ED', 'DOW', 'EMN', 'EIX', 'GIS', 'GM', 'GILD', 'HOG', 'IP', 'K', 'KSS', 'LVS', 'MPC', 'MET', 'TAP', 'JWN', 'OMC', 'PFE', 'PFG', 'O', 'REG', 'STX', 'SO', 'VTR', 'VZ', 'VNO', 'WFC', 'WDC', 'WRK']
    open       58.530645
    div_pct     4.003871
    div_amt     2.306452
    eps         3.771935
    dtype: float64
    
    cluster 18:
    includes 1 stocks
    ['AMZN']
    open       1795.02
    div_pct       0.00
    div_amt       0.00
    eps          22.57
    dtype: float64
    


# Changing our approach: The Wealthy Investor technique

I don't have too much expertise with stock trading, but have been listening to a podcast lately called *trading stocks made easy* by Tyrone Jackson (great podcast that I'd reccomend to anyone trying to learn more). He heavily advocates for stocks which pay out a dividend, a portion of their profits that isn't reinvested into the company, but rather goes to the shareholders. Additonally, he advocates for stocks that have sshowed consistent quarterly earnings growth. Between the two, dividend yield is a part of the data that has been collected, so I decided to cluster the subset of data for stocks which do pay out a dividend


```python
# get stocks which pay dividend
div_yielding_data = four_dim_cluster_data[four_dim_cluster_data['div_amt'] > 0].drop(columns=cluster_config_one.keys(), axis=1)
```


```python
k_means_func(data=div_yielding_data)
```


![png](output_28_0.png)



```python
# apply model for n = {12, 14, 19}
cluster_config_two = {
    'cluster_fourteen': 14,
    'cluster_nineteen': 19,
    'cluster_twenty_three': 23
}

div_yielding_data = classify_four_dim_stocks(div_yielding_data, cluster_config_two)
```


```python
output_cluster_tickers(original_data=df, cluster_data=div_yielding_data, cluster='cluster_twenty_three')
```

    cluster 0:
    includes 24 stocks
    ['ACN', 'APD', 'AMT', 'ADP', 'CB', 'CME', 'STZ', 'DE', 'HSY', 'HON', 'ITW', 'KLAC', 'LHX', 'LIN', 'MKC', 'MCD', 'MSI', 'NEE', 'ROK', 'SWK', 'SYK', 'UNP', 'UTX', 'WLTW']
    open       187.022500
    div_pct      1.841250
    div_amt      3.428750
    eps          6.843333
    dtype: float64
    
    cluster 1:
    includes 56 stocks
    ['AFL', 'ALK', 'ALB', 'LNT', 'AEE', 'AIG', 'AIV', 'BK', 'BMY', 'CHRW', 'CF', 'CSCO', 'C', 'CMS', 'KO', 'CL', 'COP', 'CVS', 'DAL', 'EMN', 'EMR', 'EQR', 'EVRG', 'ES', 'HIG', 'HAS', 'HFC', 'INTC', 'JCI', 'K', 'LEG', 'LNC', 'MPC', 'MXIM', 'MRK', 'MET', 'MDLZ', 'MS', 'NTAP', 'NUE', 'OMC', 'PAYX', 'PLD', 'PEG', 'QCOM', 'RHI', 'STT', 'SYF', 'SYY', 'USB', 'WBA', 'WEC', 'WFC', 'XEL', 'XRX', 'ZION']
    open       64.544643
    div_pct     2.723750
    div_amt     1.752500
    eps         3.903393
    dtype: float64
    
    cluster 2:
    includes 10 stocks
    ['MMM', 'AMGN', 'AVB', 'BA', 'ESS', 'RE', 'HD', 'IBM', 'PSA', 'SPG']
    open       222.526
    div_pct      3.329
    div_amt      6.878
    eps          8.663
    dtype: float64
    
    cluster 3:
    includes 9 stocks
    ['APA', 'COTY', 'DXC', 'KHC', 'NWL', 'NLSN', 'SLB', 'FTI', 'WDC']
    open       28.590000
    div_pct     4.234444
    div_amt     1.165556
    eps        -4.783333
    dtype: float64
    
    cluster 4:
    includes 61 stocks
    ['ATVI', 'A', 'AAL', 'AOS', 'AMAT', 'ARNC', 'BLL', 'BAC', 'BAX', 'BWA', 'CERN', 'SCHW', 'CHD', 'XEC', 'CTSH', 'CMCSA', 'CTVA', 'CSX', 'DHI', 'XRAY', 'DVN', 'DD', 'ETFC', 'EBAY', 'EXPD', 'FLIR', 'FLS', 'FBHS', 'FOXA', 'FOX', 'FCX', 'GE', 'HES', 'HRL', 'KR', 'LW', 'LEN', 'L', 'MRO', 'MAS', 'MGM', 'MOS', 'NEM', 'NWSA', 'NWS', 'NBL', 'NRG', 'ORCL', 'PNR', 'PRGO', 'PGR', 'PHM', 'PWR', 'ROL', 'SEE', 'LUV', 'TXT', 'TJX', 'WRB', 'WAB', 'XYL']
    open       48.194426
    div_pct     1.265738
    div_amt     0.588852
    eps         2.369180
    dtype: float64
    
    cluster 5:
    includes 15 stocks
    ['AAPL', 'BDX', 'CTAS', 'COO', 'COST', 'INTU', 'MKTX', 'MLM', 'MA', 'MCO', 'MSCI', 'ROP', 'SPGI', 'TFX', 'TMO']
    open       295.877333
    div_pct      0.719333
    div_amt      2.038667
    eps          8.054667
    dtype: float64
    
    cluster 6:
    includes 11 stocks
    ['MO', 'CTL', 'F', 'GPS', 'HP', 'IVZ', 'IRM', 'KIM', 'LB', 'OXY', 'WMB']
    open       25.681818
    div_pct     6.781818
    div_amt     1.770909
    eps         0.169091
    dtype: float64
    
    cluster 7:
    includes 27 stocks
    ['ARE', 'AEP', 'BXP', 'CVX', 'CLX', 'ED', 'CCI', 'DRI', 'DLR', 'DTE', 'DUK', 'ETN', 'ETR', 'EXR', 'FRT', 'GPC', 'IFF', 'SJM', 'JNJ', 'KMB', 'MAA', 'PNW', 'PG', 'TXN', 'UPS', 'VLO', 'WYNN']
    open       118.426296
    div_pct      3.174815
    div_amt      3.709259
    eps          4.370370
    dtype: float64
    
    cluster 8:
    includes 2 stocks
    ['CAH', 'NOV']
    open       37.445
    div_pct     2.210
    div_amt     1.060
    eps       -14.465
    dtype: float64
    
    cluster 9:
    includes 1 stocks
    ['AVGO']
    open       324.40
    div_pct      4.01
    div_amt     13.00
    eps          6.43
    dtype: float64
    
    cluster 10:
    includes 2 stocks
    ['BLK', 'LMT']
    open       445.285
    div_pct      2.555
    div_amt     11.400
    eps         23.465
    dtype: float64
    
    cluster 11:
    includes 25 stocks
    ['AAP', 'ALLE', 'AXP', 'AON', 'COF', 'CDW', 'CI', 'CXO', 'FANG', 'DG', 'ECL', 'EL', 'FRC', 'FTV', 'GL', 'HCA', 'IEX', 'KSU', 'NVDA', 'ODFL', 'PVH', 'UHS', 'V', 'VMC', 'DIS']
    open       147.0420
    div_pct      0.7716
    div_amt      1.1076
    eps          6.7476
    dtype: float64
    
    cluster 12:
    includes 28 stocks
    ['ABBV', 'T', 'CCL', 'D', 'DOW', 'EIX', 'XOM', 'GIS', 'GM', 'GILD', 'IP', 'KSS', 'LVS', 'TAP', 'OKE', 'PM', 'PPL', 'PFG', 'O', 'REG', 'STX', 'SLG', 'SO', 'TPR', 'VTR', 'VZ', 'WELL', 'WRK']
    open       60.232500
    div_pct     4.505714
    div_amt     2.699643
    eps         2.902500
    dtype: float64
    
    cluster 13:
    includes 1 stocks
    ['EQIX']
    open       559.60
    div_pct      1.76
    div_amt      9.84
    eps          5.91
    dtype: float64
    
    cluster 14:
    includes 10 stocks
    ['AMP', 'CAT', 'CMI', 'MTB', 'NSC', 'PH', 'PNC', 'RTN', 'SNA', 'WHR']
    open       175.944
    div_pct      2.468
    div_amt      4.241
    eps         12.811
    dtype: float64
    
    cluster 15:
    includes 1 stocks
    ['SHW']
    open       579.73
    div_pct      0.78
    div_amt      4.52
    eps         14.86
    dtype: float64
    
    cluster 16:
    includes 36 stocks
    ['AES', 'AMCR', 'ADM', 'COG', 'CPB', 'CNP', 'CFG', 'CAG', 'GLW', 'DRE', 'EXC', 'FAST', 'FITB', 'FE', 'BEN', 'HRB', 'HAL', 'HBI', 'HOG', 'HPE', 'HST', 'HPQ', 'HBAN', 'IPG', 'JNPR', 'KEY', 'KMI', 'NI', 'JWN', 'PBCT', 'PFE', 'RF', 'UDR', 'UNM', 'WU', 'WY']
    open       28.308056
    div_pct     3.518889
    div_amt     0.972778
    eps         1.740278
    dtype: float64
    
    cluster 17:
    includes 1 stocks
    ['AGN']
    open       190.50
    div_pct      1.56
    div_amt      2.96
    eps        -27.98
    dtype: float64
    
    cluster 18:
    includes 25 stocks
    ['ABT', 'AMG', 'AME', 'APH', 'APTV', 'DHR', 'EFX', 'EXPE', 'FDX', 'FIS', 'GPN', 'HLT', 'ICE', 'JKHY', 'JBHT', 'MCK', 'MCHP', 'NKE', 'PKI', 'RMD', 'ROST', 'SBAC', 'VRSK', 'ZBH', 'ZTS']
    open       126.8680
    div_pct      0.9396
    div_amt      1.1628
    eps          1.9892
    dtype: float64
    
    cluster 19:
    includes 8 stocks
    ['ANTM', 'GS', 'GWW', 'HUM', 'HII', 'LRCX', 'NOC', 'UNH']
    open       299.73875
    div_pct      1.47500
    div_amt      4.31000
    eps         16.78125
    dtype: float64
    
    cluster 20:
    includes 46 stocks
    ['ALL', 'AWK', 'ABC', 'ADI', 'AJG', 'AIZ', 'ATO', 'AVY', 'BBY', 'BR', 'CBOE', 'CE', 'CINF', 'CTXS', 'DFS', 'DOV', 'EOG', 'FMC', 'GRMN', 'IR', 'LDOS', 'LOW', 'MAR', 'MMC', 'MDT', 'MSFT', 'NDAQ', 'PCAR', 'PXD', 'PPG', 'DGX', 'RL', 'RJF', 'RSG', 'SWKS', 'SBUX', 'TGT', 'TEL', 'TIF', 'TSCO', 'TSN', 'VFC', 'WMT', 'WM', 'XLNX', 'YUM']
    open       110.026087
    div_pct      1.757174
    div_amt      1.916739
    eps          4.684783
    dtype: float64
    
    cluster 21:
    includes 2 stocks
    ['MAC', 'M']
    open       21.180
    div_pct    10.490
    div_amt     2.255
    eps         1.835
    dtype: float64
    



```python
other_keys = [key for key in cluster_config_two.keys() if key != 'cluster_twenty_three']
div_yielding_agg = div_yielding_data.drop(columns=other_keys, axis=1).groupby('cluster_twenty_three').mean()
```


```python
div_yielding_agg
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eps</th>
      <th>open</th>
      <th>div_pct</th>
      <th>div_amt</th>
    </tr>
    <tr>
      <th>cluster_twenty_three</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.843333</td>
      <td>187.022500</td>
      <td>1.841250</td>
      <td>3.428750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.903393</td>
      <td>64.544643</td>
      <td>2.723750</td>
      <td>1.752500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.663000</td>
      <td>222.526000</td>
      <td>3.329000</td>
      <td>6.878000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-4.783333</td>
      <td>28.590000</td>
      <td>4.234444</td>
      <td>1.165556</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.369180</td>
      <td>48.194426</td>
      <td>1.265738</td>
      <td>0.588852</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.054667</td>
      <td>295.877333</td>
      <td>0.719333</td>
      <td>2.038667</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.169091</td>
      <td>25.681818</td>
      <td>6.781818</td>
      <td>1.770909</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.370370</td>
      <td>118.426296</td>
      <td>3.174815</td>
      <td>3.709259</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-14.465000</td>
      <td>37.445000</td>
      <td>2.210000</td>
      <td>1.060000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6.430000</td>
      <td>324.400000</td>
      <td>4.010000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>23.465000</td>
      <td>445.285000</td>
      <td>2.555000</td>
      <td>11.400000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6.747600</td>
      <td>147.042000</td>
      <td>0.771600</td>
      <td>1.107600</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2.902500</td>
      <td>60.232500</td>
      <td>4.505714</td>
      <td>2.699643</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5.910000</td>
      <td>559.600000</td>
      <td>1.760000</td>
      <td>9.840000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>12.811000</td>
      <td>175.944000</td>
      <td>2.468000</td>
      <td>4.241000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>14.860000</td>
      <td>579.730000</td>
      <td>0.780000</td>
      <td>4.520000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.740278</td>
      <td>28.308056</td>
      <td>3.518889</td>
      <td>0.972778</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-27.980000</td>
      <td>190.500000</td>
      <td>1.560000</td>
      <td>2.960000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.989200</td>
      <td>126.868000</td>
      <td>0.939600</td>
      <td>1.162800</td>
    </tr>
    <tr>
      <th>19</th>
      <td>16.781250</td>
      <td>299.738750</td>
      <td>1.475000</td>
      <td>4.310000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>4.684783</td>
      <td>110.026087</td>
      <td>1.757174</td>
      <td>1.916739</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.835000</td>
      <td>21.180000</td>
      <td>10.490000</td>
      <td>2.255000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>9.195333</td>
      <td>114.239333</td>
      <td>2.994667</td>
      <td>3.286000</td>
    </tr>
  </tbody>
</table>
</div>



# Plotting the results

Finally! We have some simple visualization of the aggregated data for our clustered dividend yielding S&P 500 stocks. Based on these plots, I'm going to take a closer look at a few of the clusters:

1. cluster 10/19: these clusters has the highest earnings per share on average of all clusters
2. cluster 9/10/13: These clusters had the highest average dividend
   amounts per share of any cluster
3. cluster 6/21: these clusters by far had the highest percentage dividend of any cluster

Although open value was included in the feature set (with the intention of clustering stocks based on similar cost per share), open value for an arbritrary day does not seem like a good feature to indicate a specific cluster to consider more carefully


```python
plt.figure(figsize=(12,12))
ax1 = plt.subplot(221)
ax1.title.set_text('average EPS per cluster')
sns.barplot(x=div_yielding_agg.index, y=div_yielding_agg.eps)
ax2 = plt.subplot(222)
ax2.title.set_text('average dividend amount per cluster')
sns.barplot(x=div_yielding_agg.index, y=div_yielding_agg.div_amt)
ax3 = plt.subplot(223)
ax3.title.set_text('average dividend percentage per cluster')
sns.barplot(x=div_yielding_agg.index, y=div_yielding_agg.div_pct)
ax4 = plt.subplot(224)
ax4.title.set_text('average open value per cluster')
sns.barplot(x=div_yielding_agg.index, y=div_yielding_agg.open)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1fe5d630>




![png](output_34_1.png)


# Results

Although these results are far from finished, and I will need to comb through financial figures and track these
stocks for more than just one day, it is clear that clustering through the K-means algorithm has allowed me to hone
initial search for potentially lucrative S&P 500 stocks. This was a fun and quick 1-day venture that allowed me to
get more familiar with relevant financial figures for stock trading, scraping stock data, and applying machine
learning techniques to an interesting data set


```python
# we can use the output cluster tickers function, passsing an optional parameter which specifies
# which clusters to show the tickers for.
output_cluster_tickers(original_data=df, cluster_data=div_yielding_data, cluster='cluster_twenty_three', show_tickers=[6, 9, 10, 13, 19, 21])
```

    cluster 6:
    includes 11 stocks
    ['MO', 'CTL', 'F', 'GPS', 'HP', 'IVZ', 'IRM', 'KIM', 'LB', 'OXY', 'WMB']
    open       25.681818
    div_pct     6.781818
    div_amt     1.770909
    eps         0.169091
    dtype: float64
    
    cluster 9:
    includes 1 stocks
    ['AVGO']
    open       324.40
    div_pct      4.01
    div_amt     13.00
    eps          6.43
    dtype: float64
    
    cluster 10:
    includes 2 stocks
    ['BLK', 'LMT']
    open       445.285
    div_pct      2.555
    div_amt     11.400
    eps         23.465
    dtype: float64
    
    cluster 13:
    includes 1 stocks
    ['EQIX']
    open       559.60
    div_pct      1.76
    div_amt      9.84
    eps          5.91
    dtype: float64
    
    cluster 19:
    includes 8 stocks
    ['ANTM', 'GS', 'GWW', 'HUM', 'HII', 'LRCX', 'NOC', 'UNH']
    open       299.73875
    div_pct      1.47500
    div_amt      4.31000
    eps         16.78125
    dtype: float64
    
    cluster 21:
    includes 2 stocks
    ['MAC', 'M']
    open       21.180
    div_pct    10.490
    div_amt     2.255
    eps         1.835
    dtype: float64
    



```python
# we can use the output cluster tickers function, passsing an optional parameter which specifies
# which clusters to show the tickers for.
output_cluster_tickers(original_data=df, cluster_data=div_yielding_data, cluster='cluster_nineteen')
```

    cluster 0:
    includes 17 stocks
    ['AAPL', 'BDX', 'CI', 'CTAS', 'COO', 'COST', 'INTU', 'MKTX', 'MLM', 'MA', 'MCO', 'MSCI', 'ROP', 'SPGI', 'SYK', 'TFX', 'TMO']
    open       284.572941
    div_pct      0.702353
    div_amt      1.936471
    eps          8.352353
    dtype: float64
    
    cluster 1:
    includes 58 stocks
    ['ALK', 'ALB', 'LNT', 'AEE', 'AEP', 'AIV', 'ADM', 'BK', 'BMY', 'CHRW', 'CSCO', 'C', 'CFG', 'CMS', 'KO', 'CL', 'COP', 'CVS', 'DAL', 'EMN', 'EMR', 'EQR', 'EVRG', 'ES', 'EXC', 'FITB', 'FE', 'GIS', 'HIG', 'HAS', 'HFC', 'INTC', 'JCI', 'K', 'LEG', 'LNC', 'MPC', 'MXIM', 'MRK', 'MET', 'MS', 'NTAP', 'NUE', 'OMC', 'PAYX', 'PFG', 'PLD', 'PEG', 'QCOM', 'STT', 'SYF', 'SYY', 'USB', 'WBA', 'WEC', 'WFC', 'XEL', 'ZION']
    open       64.173103
    div_pct     2.854655
    div_amt     1.809828
    eps         3.905172
    dtype: float64
    
    cluster 2:
    includes 7 stocks
    ['AMP', 'CMI', 'GS', 'MTB', 'SNA', 'VNO', 'WHR']
    open       162.411429
    div_pct      2.827143
    div_amt      4.325714
    eps         15.898571
    dtype: float64
    
    cluster 3:
    includes 32 stocks
    ['ARE', 'ADS', 'BXP', 'CVX', 'CLX', 'CMA', 'ED', 'DRI', 'DTE', 'ETN', 'ETR', 'FRT', 'GPC', 'HSY', 'SJM', 'JNJ', 'KMB', 'LLY', 'LYB', 'MAA', 'NTRS', 'PKG', 'PEP', 'PSX', 'PRU', 'RCL', 'TROW', 'TXN', 'TRV', 'UPS', 'VLO', 'WYNN']
    open       119.817812
    div_pct      3.031875
    div_amt      3.562812
    eps          6.331563
    dtype: float64
    
    cluster 4:
    includes 8 stocks
    ['APA', 'CAH', 'CTL', 'COTY', 'KHC', 'NLSN', 'SLB', 'WDC']
    open       30.72875
    div_pct     4.91375
    div_amt     1.39125
    eps        -6.71250
    dtype: float64
    
    cluster 5:
    includes 4 stocks
    ['BA', 'AVGO', 'EQIX', 'ESS']
    open       377.2375
    div_pct      2.7275
    div_amt      9.7150
    eps          6.3675
    dtype: float64
    
    cluster 6:
    includes 56 stocks
    ['ALL', 'AXP', 'AWK', 'ABC', 'ADI', 'AJG', 'AIZ', 'ATO', 'AVY', 'BBY', 'BR', 'COF', 'CBOE', 'CE', 'CINF', 'CTXS', 'STZ', 'DFS', 'DOV', 'EXPE', 'FDX', 'FMC', 'GRMN', 'IR', 'IFF', 'LDOS', 'LOW', 'MAR', 'MMC', 'MKC', 'MDT', 'MCHP', 'MSFT', 'MSI', 'NDAQ', 'PCAR', 'PPG', 'PG', 'DGX', 'RL', 'RJF', 'RSG', 'SWKS', 'SWK', 'SBUX', 'TGT', 'TEL', 'TIF', 'TSCO', 'TSN', 'UTX', 'VFC', 'WMT', 'WM', 'XLNX', 'YUM']
    open       116.192143
    div_pct      1.759286
    div_amt      2.030893
    eps          4.663929
    dtype: float64
    
    cluster 7:
    includes 50 stocks
    ['ATVI', 'A', 'AAL', 'AME', 'APH', 'AMAT', 'APTV', 'ARNC', 'BLL', 'BAX', 'BWA', 'CERN', 'SCHW', 'CHD', 'XEC', 'CTSH', 'CXO', 'CSX', 'DHI', 'XRAY', 'DVN', 'FANG', 'ETFC', 'EOG', 'EXPD', 'FLIR', 'FTV', 'FBHS', 'FOXA', 'FOX', 'GE', 'HLT', 'ICE', 'LW', 'LEN', 'L', 'MAS', 'NEM', 'NKE', 'NRG', 'PKI', 'PGR', 'PHM', 'PWR', 'LUV', 'TXT', 'TJX', 'WRB', 'WAB', 'XYL']
    open       63.3910
    div_pct     0.9736
    div_amt     0.6050
    eps         3.2894
    dtype: float64
    
    cluster 8:
    includes 1 stocks
    ['AGN']
    open       190.50
    div_pct      1.56
    div_amt      2.96
    eps        -27.98
    dtype: float64
    
    cluster 9:
    includes 2 stocks
    ['BLK', 'LMT']
    open       445.285
    div_pct      2.555
    div_amt     11.400
    eps         23.465
    dtype: float64
    
    cluster 10:
    includes 31 stocks
    ['AAP', 'ALLE', 'AON', 'CDW', 'DHR', 'DG', 'ECL', 'EL', 'FIS', 'FRC', 'GL', 'GPN', 'HCA', 'IEX', 'JKHY', 'JBHT', 'KSU', 'NVDA', 'ODFL', 'PXD', 'PVH', 'RMD', 'ROST', 'SBAC', 'UHS', 'VRSK', 'V', 'VMC', 'DIS', 'ZBH', 'ZTS']
    open       155.143548
    div_pct      0.778065
    div_amt      1.189677
    eps          4.845484
    dtype: float64
    
    cluster 11:
    includes 10 stocks
    ['MO', 'F', 'HP', 'IVZ', 'IRM', 'LB', 'MAC', 'M', 'OXY', 'WMB']
    open       27.418
    div_pct     7.692
    div_amt     2.090
    eps         1.003
    dtype: float64
    
    cluster 12:
    includes 41 stocks
    ['ABT', 'AES', 'AFL', 'AIG', 'AOS', 'BAC', 'COG', 'CPB', 'CF', 'CMCSA', 'CAG', 'GLW', 'CTVA', 'DRE', 'DD', 'EBAY', 'FAST', 'FLS', 'FCX', 'HAL', 'HES', 'HPE', 'HRL', 'JNPR', 'KR', 'MRO', 'MGM', 'MDLZ', 'MOS', 'NWSA', 'NWS', 'NI', 'ORCL', 'PNR', 'PRGO', 'RHI', 'ROL', 'SEE', 'UDR', 'WU', 'XRX']
    open       37.565122
    div_pct     2.142927
    div_amt     0.790732
    eps         1.632439
    dtype: float64
    
    cluster 13:
    includes 10 stocks
    ['AMGN', 'ANTM', 'RE', 'GWW', 'HUM', 'HII', 'LRCX', 'NOC', 'SHW', 'UNH']
    open       326.680
    div_pct      1.528
    div_amt      4.660
    eps         15.004
    dtype: float64
    
    cluster 14:
    includes 27 stocks
    ['MMM', 'ACN', 'APD', 'AMT', 'ADP', 'AVB', 'CAT', 'CB', 'CME', 'DE', 'HD', 'HON', 'ITW', 'JPM', 'KLAC', 'LHX', 'LIN', 'MCD', 'NEE', 'NSC', 'PH', 'PNC', 'RTN', 'ROK', 'SRE', 'UNP', 'WLTW']
    open       189.472593
    div_pct      2.135185
    div_amt      3.988148
    eps          8.268148
    dtype: float64
    
    cluster 15:
    includes 23 stocks
    ['ABBV', 'CCI', 'DLR', 'D', 'DOW', 'DUK', 'EIX', 'EXR', 'XOM', 'GILD', 'KSS', 'LVS', 'OKE', 'PM', 'PNW', 'O', 'REG', 'STX', 'SLG', 'SO', 'VTR', 'VZ', 'WELL']
    open       77.526087
    div_pct     4.353913
    div_amt     3.316087
    eps         2.796087
    dtype: float64
    
    cluster 16:
    includes 3 stocks
    ['IBM', 'PSA', 'SPG']
    open       161.196667
    div_pct      4.833333
    div_amt      7.626667
    eps          8.170000
    dtype: float64
    
    cluster 17:
    includes 29 stocks
    ['AMCR', 'T', 'CCL', 'CNP', 'BEN', 'GPS', 'GM', 'HRB', 'HBI', 'HOG', 'HST', 'HPQ', 'HBAN', 'IP', 'IPG', 'KEY', 'KIM', 'KMI', 'TAP', 'NWL', 'JWN', 'PBCT', 'PFE', 'PPL', 'RF', 'TPR', 'UNM', 'WRK', 'WY']
    open       27.975517
    div_pct     4.359310
    div_amt     1.215172
    eps         2.048621
    dtype: float64
    

