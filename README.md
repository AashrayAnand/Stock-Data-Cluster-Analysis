# S&P 500 Stock Clustering

credit: https://github.com/ttimong/blog-posts/blob/master/blog1-kmeans-clustering/final_model.ipynb (for K-means metrics function)

This notebook demonstrates a clustering of the S&P 500 stock exchange, based on a select set of financial figures

The exchange consists of 500 companies, but includes 505 common stocks, due to 5 companies having two shares of stocks in the exchange (Facebook, Under-Armour, NewsCorp, Comcast and 21st Century Fox)


```python
# libraries for making requests and parsing HTML
import requests
from bs4 import BeautifulSoup

# plotting
import matplotlib.pyplot as plt

# sklearn for kmeans and model metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

# pandas, for data wrangling
import pandas as pd
```

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
df = df[df['eps'] != 'N/A']
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

    /anaconda3/envs/machine_learning/lib/python3.6/site-packages/pandas/core/ops/__init__.py:1115: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      result = method(y)





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




    <matplotlib.axes._subplots.AxesSubplot at 0x1a359fc780>




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
    transform_data = RobustScaler().fit_transform(data.iloc[:,1:])
    
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
    transform_data = RobustScaler().fit_transform(data.iloc[:,1:])
    # initialize K-means models with each of the specified cluster hyperparameter valuess
    for config in cluster_configs.keys():
        model = KMeans(n_clusters=cluster_configs[config], random_state=5, n_init=10)
        model.fit(transform_data)
        data[config] = model.labels_
    return data
```


```python
cluster_config = {
    'cluster_five': 5,
    'cluster_ten': 10,
    'cluster_fourteen': 14,
    'cluster_twenty': 20
}
four_dim_cluster_data = classify_four_dim_stocks(four_dim_cluster_data[['ticker', 'eps', 'open', 'div_pct', 'div_amt']], cluster_config)
```


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
      <td>4</td>
      <td>3</td>
      <td>6</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ABT</td>
      <td>1.84</td>
      <td>87.08</td>
      <td>1.65</td>
      <td>1.44</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABBV</td>
      <td>2.18</td>
      <td>90.05</td>
      <td>5.24</td>
      <td>4.72</td>
      <td>0</td>
      <td>9</td>
      <td>9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABMD</td>
      <td>4.79</td>
      <td>179.85</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3</td>
      <td>7</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>7.36</td>
      <td>203.60</td>
      <td>1.83</td>
      <td>3.72</td>
      <td>4</td>
      <td>6</td>
      <td>6</td>
      <td>17</td>
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
      <th>499</th>
      <td>XYL</td>
      <td>2.80</td>
      <td>78.10</td>
      <td>1.23</td>
      <td>0.96</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>500</th>
      <td>YUM</td>
      <td>3.62</td>
      <td>99.48</td>
      <td>1.69</td>
      <td>1.68</td>
      <td>3</td>
      <td>0</td>
      <td>11</td>
      <td>9</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZBH</td>
      <td>-0.44</td>
      <td>149.90</td>
      <td>0.64</td>
      <td>0.96</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZION</td>
      <td>4.27</td>
      <td>51.60</td>
      <td>2.64</td>
      <td>1.36</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>503</th>
      <td>ZTS</td>
      <td>3.02</td>
      <td>127.15</td>
      <td>0.63</td>
      <td>0.80</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>497 rows × 9 columns</p>
</div>




```python
def output_cluster_tickers(original_data, cluster_data, cluster, show_tickers=[]): 
    for i in range(0, max(cluster_data[cluster])):
        # list of tickers for the current cluster
        ticker_list = list(cluster_data[cluster_data[cluster] == i]['ticker'])
        print("cluster " + str(i) + ":")
        print("includes " + str(len(ticker_list)) + " stocks")
        if(i in show_tickers):
            print(ticker_list)
        # original data for tickers that are part of cluster, more useful than
        # the transformed data
        curr_data = original_data[original_data['ticker'].isin(ticker_list)]
        print(curr_data[['open', 'div_pct', 'div_amt', 'eps']].mean())
        print()
```


```python
output_clusters(df, four_dim_cluster_data, 'cluster_twenty')
```

    cluster 0:
    includes 28 stocks
    open       150.557500
    div_pct      0.282143
    div_amt      0.350357
    eps          7.986429
    dtype: float64
    
    cluster 1:
    includes 1 stocks
    open       3820.00
    div_pct       0.00
    div_amt       0.00
    eps         215.31
    dtype: float64
    
    cluster 2:
    includes 101 stocks
    open       64.763663
    div_pct     0.686238
    div_amt     0.360495
    eps         2.357624
    dtype: float64
    
    cluster 3:
    includes 1 stocks
    open       2008.67
    div_pct       0.00
    div_amt       0.00
    eps          97.36
    dtype: float64
    
    cluster 4:
    includes 1 stocks
    open       1795.02
    div_pct       0.00
    div_amt       0.00
    eps          22.57
    dtype: float64
    
    cluster 5:
    includes 22 stocks
    open       27.763636
    div_pct     6.433636
    div_amt     1.762727
    eps         0.315455
    dtype: float64
    
    cluster 6:
    includes 43 stocks
    open       102.298837
    div_pct      3.685349
    div_amt      3.637209
    eps          4.666047
    dtype: float64
    
    cluster 7:
    includes 5 stocks
    open       346.014
    div_pct      0.120
    div_amt      0.440
    eps         20.878
    dtype: float64
    
    cluster 8:
    includes 3 stocks
    open       1311.956667
    div_pct       0.000000
    div_amt       0.000000
    eps          52.210000
    dtype: float64
    
    cluster 9:
    includes 85 stocks
    open       117.334706
    div_pct      1.622118
    div_amt      1.808706
    eps          4.571294
    dtype: float64
    
    cluster 10:
    includes 14 stocks
    open       250.187857
    div_pct      2.015714
    div_amt      4.734286
    eps         15.101429
    dtype: float64
    
    cluster 11:
    includes 5 stocks
    open       670.012
    div_pct      0.156
    div_amt      0.904
    eps         14.500
    dtype: float64
    
    cluster 12:
    includes 1 stocks
    open       190.50
    div_pct      1.56
    div_amt      2.96
    eps        -27.98
    dtype: float64
    
    cluster 13:
    includes 97 stocks
    open       48.254021
    div_pct     3.142887
    div_amt     1.468557
    eps         2.751856
    dtype: float64
    
    cluster 14:
    includes 2 stocks
    open       445.285
    div_pct      2.555
    div_amt     11.400
    eps         23.465
    dtype: float64
    
    cluster 15:
    includes 1 stocks
    open       135.73
    div_pct      0.00
    div_amt      0.00
    eps         49.90
    dtype: float64
    
    cluster 16:
    includes 28 stocks
    open       292.122857
    div_pct      0.410000
    div_amt      1.120714
    eps          6.551786
    dtype: float64
    
    cluster 17:
    includes 40 stocks
    open       171.01675
    div_pct      2.27775
    div_amt      3.74425
    eps          8.56550
    dtype: float64
    
    cluster 18:
    includes 6 stocks
    open       309.665000
    div_pct      3.436667
    div_amt      9.210000
    eps          6.896667
    dtype: float64
    


# Changing our approach: The Wealthy Investor technique

I don't have too much expertise with stock trading, but have been listening to a podcast lately called *trading stocks made easy* by Tyrone Jackson (great podcast that I'd reccomend to anyone trying to learn more). He heavily advocates for stocks which pay out a dividend, a portion of their profits that isn't reinvested into the company, but rather goes to the shareholders. Additonally, he advocates for stocks that have sshowed consistent quarterly earnings growth. Between the two, dividend yield is a part of the data that has been collected, so I decided to cluster the subset of data for stocks which do pay out a dividend


```python
# get stocks which pay dividend
div_yielding_data = four_dim_cluster_data[four_dim_cluster_data['div_amt'] > 0]
```


```python
k_means_func(data=div_yielding_data)
```


![png](output_28_0.png)



```python
# apply model for n = {12, 14, 19}
cluster_config = {
    'cluster_twelve': 12,
    'cluster_fourteen': 14,
    'cluster_nineteen': 19
}
div_yielding_data = classify_four_dim_stocks(div_yielding_data, cluster_config)
```

    /anaconda3/envs/machine_learning/lib/python3.6/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys
    /anaconda3/envs/machine_learning/lib/python3.6/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys
    /anaconda3/envs/machine_learning/lib/python3.6/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys



```python
output_clusters(original_data=df, cluster_data=div_yielding_data, cluster='cluster_twenty')
```

    cluster 0:
    includes 13 stocks
    open       132.898462
    div_pct      0.607692
    div_amt      0.754615
    eps          7.920769
    dtype: float64
    
    cluster 1:
    includes 0 stocks
    open      NaN
    div_pct   NaN
    div_amt   NaN
    eps       NaN
    dtype: float64
    
    cluster 2:
    includes 62 stocks
    open       56.385484
    div_pct     1.117903
    div_amt     0.587258
    eps         2.474677
    dtype: float64
    
    cluster 3:
    includes 0 stocks
    open      NaN
    div_pct   NaN
    div_amt   NaN
    eps       NaN
    dtype: float64
    
    cluster 4:
    includes 0 stocks
    open      NaN
    div_pct   NaN
    div_amt   NaN
    eps       NaN
    dtype: float64
    
    cluster 5:
    includes 22 stocks
    open       27.763636
    div_pct     6.433636
    div_amt     1.762727
    eps         0.315455
    dtype: float64
    
    cluster 6:
    includes 43 stocks
    open       102.298837
    div_pct      3.685349
    div_amt      3.637209
    eps          4.666047
    dtype: float64
    
    cluster 7:
    includes 1 stocks
    open       363.92
    div_pct      0.60
    div_amt      2.20
    eps         18.78
    dtype: float64
    
    cluster 8:
    includes 0 stocks
    open      NaN
    div_pct   NaN
    div_amt   NaN
    eps       NaN
    dtype: float64
    
    cluster 9:
    includes 85 stocks
    open       117.334706
    div_pct      1.622118
    div_amt      1.808706
    eps          4.571294
    dtype: float64
    
    cluster 10:
    includes 14 stocks
    open       250.187857
    div_pct      2.015714
    div_amt      4.734286
    eps         15.101429
    dtype: float64
    
    cluster 11:
    includes 1 stocks
    open       579.73
    div_pct      0.78
    div_amt      4.52
    eps         14.86
    dtype: float64
    
    cluster 12:
    includes 1 stocks
    open       190.50
    div_pct      1.56
    div_amt      2.96
    eps        -27.98
    dtype: float64
    
    cluster 13:
    includes 97 stocks
    open       48.254021
    div_pct     3.142887
    div_amt     1.468557
    eps         2.751856
    dtype: float64
    
    cluster 14:
    includes 2 stocks
    open       445.285
    div_pct      2.555
    div_amt     11.400
    eps         23.465
    dtype: float64
    
    cluster 15:
    includes 0 stocks
    open      NaN
    div_pct   NaN
    div_amt   NaN
    eps       NaN
    dtype: float64
    
    cluster 16:
    includes 17 stocks
    open       283.738235
    div_pct      0.675294
    div_amt      1.845882
    eps          7.070000
    dtype: float64
    
    cluster 17:
    includes 40 stocks
    open       171.01675
    div_pct      2.27775
    div_amt      3.74425
    eps          8.56550
    dtype: float64
    
    cluster 18:
    includes 6 stocks
    open       309.665000
    div_pct      3.436667
    div_amt      9.210000
    eps          6.896667
    dtype: float64
    



```python
div_yielding_agg = div_yielding_data.drop(columns=['cluster_five', 'cluster_ten', 'cluster_fourteen'], axis=1).groupby('cluster_twenty').mean()
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
      <th>cluster_twelve</th>
      <th>cluster_nineteen</th>
    </tr>
    <tr>
      <th>cluster_twenty</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.920769</td>
      <td>132.898462</td>
      <td>0.607692</td>
      <td>0.754615</td>
      <td>4.538462</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.474677</td>
      <td>56.385484</td>
      <td>1.117903</td>
      <td>0.587258</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.315455</td>
      <td>27.763636</td>
      <td>6.433636</td>
      <td>1.762727</td>
      <td>8.727273</td>
      <td>13.818182</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.666047</td>
      <td>102.298837</td>
      <td>3.685349</td>
      <td>3.637209</td>
      <td>3.046512</td>
      <td>10.395349</td>
    </tr>
    <tr>
      <th>7</th>
      <td>18.780000</td>
      <td>363.920000</td>
      <td>0.600000</td>
      <td>2.200000</td>
      <td>6.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.571294</td>
      <td>117.334706</td>
      <td>1.622118</td>
      <td>1.808706</td>
      <td>5.882353</td>
      <td>9.317647</td>
    </tr>
    <tr>
      <th>10</th>
      <td>15.101429</td>
      <td>250.187857</td>
      <td>2.015714</td>
      <td>4.734286</td>
      <td>6.285714</td>
      <td>1.928571</td>
    </tr>
    <tr>
      <th>11</th>
      <td>14.860000</td>
      <td>579.730000</td>
      <td>0.780000</td>
      <td>4.520000</td>
      <td>6.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-27.980000</td>
      <td>190.500000</td>
      <td>1.560000</td>
      <td>2.960000</td>
      <td>7.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2.751856</td>
      <td>48.254021</td>
      <td>3.142887</td>
      <td>1.468557</td>
      <td>0.958763</td>
      <td>0.556701</td>
    </tr>
    <tr>
      <th>14</th>
      <td>23.465000</td>
      <td>445.285000</td>
      <td>2.555000</td>
      <td>11.400000</td>
      <td>4.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>7.070000</td>
      <td>283.738235</td>
      <td>0.675294</td>
      <td>1.845882</td>
      <td>10.000000</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>8.565500</td>
      <td>171.016750</td>
      <td>2.277750</td>
      <td>3.744250</td>
      <td>1.675000</td>
      <td>5.675000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6.896667</td>
      <td>309.665000</td>
      <td>3.436667</td>
      <td>9.210000</td>
      <td>11.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-6.420000</td>
      <td>54.525000</td>
      <td>2.789167</td>
      <td>1.211667</td>
      <td>6.666667</td>
      <td>10.666667</td>
    </tr>
  </tbody>
</table>
</div>



# Plotting the results

Finally! We have some simple visualization of the aggregated data for our clustered dividend yielding S&P 500 stocks. Based on these plots, I'm going to take a closer look at a few of the clusters:

1. cluster 14: this cluster has the highest earnings per share on average of all clusters
2. cluster 18: This cluster (along with the aforementioned cluster 14) has one of highest average dividend
   amounts per share of any cluster
3. cluster 5: this cluster by far has the highest percentage dividend of any cluster

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




    <matplotlib.axes._subplots.AxesSubplot at 0x1a3d34fe80>




![png](output_34_1.png)



```python
# we can use the output cluster tickers function, passsing an optional parameter which specifies
# which clusters to show the tickers for.
output_cluster_tickers(original_data=df, cluster_data=div_yielding_data, cluster='cluster_twenty', show_tickers=[5, 14, 18])
```

    cluster 0:
    includes 13 stocks
    open       132.898462
    div_pct      0.607692
    div_amt      0.754615
    eps          7.920769
    dtype: float64
    
    cluster 1:
    includes 0 stocks
    open      NaN
    div_pct   NaN
    div_amt   NaN
    eps       NaN
    dtype: float64
    
    cluster 2:
    includes 62 stocks
    open       56.385484
    div_pct     1.117903
    div_amt     0.587258
    eps         2.474677
    dtype: float64
    
    cluster 3:
    includes 0 stocks
    open      NaN
    div_pct   NaN
    div_amt   NaN
    eps       NaN
    dtype: float64
    
    cluster 4:
    includes 0 stocks
    open      NaN
    div_pct   NaN
    div_amt   NaN
    eps       NaN
    dtype: float64
    
    cluster 5:
    includes 22 stocks
    ['MO', 'APA', 'T', 'CTL', 'DOW', 'F', 'GPS', 'HP', 'IVZ', 'IRM', 'KIM', 'KMI', 'LB', 'MAC', 'M', 'NWL', 'NLSN', 'OXY', 'TPR', 'VTR', 'WY', 'WMB']
    open       27.763636
    div_pct     6.433636
    div_amt     1.762727
    eps         0.315455
    dtype: float64
    
    cluster 6:
    includes 43 stocks
    open       102.298837
    div_pct      3.685349
    div_amt      3.637209
    eps          4.666047
    dtype: float64
    
    cluster 7:
    includes 1 stocks
    open       363.92
    div_pct      0.60
    div_amt      2.20
    eps         18.78
    dtype: float64
    
    cluster 8:
    includes 0 stocks
    open      NaN
    div_pct   NaN
    div_amt   NaN
    eps       NaN
    dtype: float64
    
    cluster 9:
    includes 85 stocks
    open       117.334706
    div_pct      1.622118
    div_amt      1.808706
    eps          4.571294
    dtype: float64
    
    cluster 10:
    includes 14 stocks
    open       250.187857
    div_pct      2.015714
    div_amt      4.734286
    eps         15.101429
    dtype: float64
    
    cluster 11:
    includes 1 stocks
    open       579.73
    div_pct      0.78
    div_amt      4.52
    eps         14.86
    dtype: float64
    
    cluster 12:
    includes 1 stocks
    open       190.50
    div_pct      1.56
    div_amt      2.96
    eps        -27.98
    dtype: float64
    
    cluster 13:
    includes 97 stocks
    open       48.254021
    div_pct     3.142887
    div_amt     1.468557
    eps         2.751856
    dtype: float64
    
    cluster 14:
    includes 2 stocks
    ['BLK', 'LMT']
    open       445.285
    div_pct      2.555
    div_amt     11.400
    eps         23.465
    dtype: float64
    
    cluster 15:
    includes 0 stocks
    open      NaN
    div_pct   NaN
    div_amt   NaN
    eps       NaN
    dtype: float64
    
    cluster 16:
    includes 17 stocks
    open       283.738235
    div_pct      0.675294
    div_amt      1.845882
    eps          7.070000
    dtype: float64
    
    cluster 17:
    includes 40 stocks
    open       171.01675
    div_pct      2.27775
    div_amt      3.74425
    eps          8.56550
    dtype: float64
    
    cluster 18:
    includes 6 stocks
    ['BA', 'AVGO', 'EQIX', 'ESS', 'PSA', 'SPG']
    open       309.665000
    div_pct      3.436667
    div_amt      9.210000
    eps          6.896667
    dtype: float64
    


# Results

Although these results are far from finished, and I will need to comb through financial figures and track these
stocks for more than just one day, it is clear that clustering through the K-means algorithm has allowed me to hone
initial search for potentially lucrative S&P 500 stocks. This was a fun and quick 1-day venture that allowed me to
get more familiar with relevant financial figures for stock trading, scraping stock data, and applying machine
learning techniques to an interesting data set

*cluster five*:
1. MO
2. APA
3. T
4. CTL
5. DOW
6. F
7. GPS
8. HP
9. IVZ
10. IRM
11. KIM
12. KMI
13. LB
14. MAC
15. M
16. NWL
17. NLSN
18. OXY
19. TPR
20. VTR
21. WY
22. WMB

*cluster fourteen*:
1. BLK
2. LMT

*cluster eighteen:
1. BA
2. AVGO
3. EQIX
4. ESS
5. PSA
6. SPG


```python

```
