
# coding: utf-8

# this notebook will download all the data needed for the calculations. The main source is AlphaVantage, which has a curated collection of adjusted  stock prices that are freely downloadable.
# 
# If you want to download the data, please go through each cell carefully, and mind the comments. This is not meant to be really stable and tested code, and requires your attention to be run properly.

# In[ ]:


from urllib import request
from urllib.error import HTTPError
from io import StringIO
import pandas as pd
import numpy as np
from time import sleep
from tqdm import tqdm
import pickle
import sys
sys.path.append('../../')
from modelinter.constants import Paths


# In[ ]:


#we'll download and save the list of tickers included in the S&P500 as of today
# http://data.okfn.org/data/core/s-and-p-500-companies#readme

constituents_csv = Paths.datadir.value + 'sp500_constituents.csv'
constituents_url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
with open(constituents_csv, 'w') as file:
    file.write(request.urlopen(constituents_url).read().decode())


# In[ ]:


constituents = pd.read_csv(constituents_csv)


# In[ ]:


#read the api key from alpha vantage. you need to put your API key here:
##      ../data/raw/alphavantage_apikey.txt
#or else the code won't work
apikey = open(Paths.datadir.value+'alphavantage_apikey.txt', 'r').read().strip()
#head and tail of the HTML query we're going to use to download each timeseries
head = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='
tail = '&apikey='+apikey+'&datatype=csv&outputsize=full'


# In[ ]:


#query the API with the string "req"
#and return a dataframe with the contents
#of the CSV that is sent in response.
def query_API(req):
    data = request.urlopen(req).read()
    df = pd.read_csv(StringIO(data.decode()))
    return(df)


# In[ ]:


#we'll put temporary results here
timeseries = []
#and list the tickets that had errors here
httperrors = []
othererrors = []


# In[ ]:


#list of tickers to download
tickers = constituents.Symbol.tolist()[:]

#when you're done running the loop below, you can run the line below
#which will add the tickers that were not downloaded to the list "tickers"
#and then re-run the loop to download the missing tickers.
tickers = list(set(othererrors)) + list(set(httperrors))
# In[ ]:


#this is where we'll save the temp results, in case of crashes during execution.
temp_timeseries_path = Paths.savedir.value + 'temp_timeseries.pkl'


# In[ ]:


#this loop downloads the tickers contained in the list "tickers"
#and puts the data in the list "timeseries".
#it's a bit rudimentary, but if you don't need fancy stuff,
#it's better not to use it.
i,tot = 0, len(tickers)
while tickers: #this is so if it breaks you can resume
    #select a ticker to download
    ticker = tickers[-1]
    print(i/tot, end='   ')
    i+=1
    #build API call
    req = head+ticker+tail
    try:
        #download
        df = query_API(req)
        #throw away other columns of the dataframe
        #we only need adjusted close.
        df = df[['timestamp','adjusted_close']]
    except KeyError as error:
        #if ticker missing from API
        #tell me there's an error
        tqdm.write('ERROR ON:' + ticker + str(error))
        #save in list "othererrors"
        othererrors.append(ticker)
        #pop from list
        tickers.pop()
        continue
    except AttributeError as error:
        tqdm.write('ERROR ON:' + ticker + str(error))
        othererrors.append(ticker)
        tickers.pop()
        continue
    except HTTPError as error:
        #HTTP error usually happens because
        #the connection has trouble, so retry
        #instead of skipping
        tqdm.write('ERROR ON:' + ticker + str(error))
        sleep(5)
        httperrors.append(ticker)
        #don't pop tickers, retry in 5 seconds
        continue
    #now we downloaded the ticker.
    #convert timestamp to pandas format
    df.timestamp = pd.to_datetime(df.timestamp, format='%Y-%m-%d')
    df = df.set_index('timestamp')
    #rename (only) column
    df.columns = [ticker]
    timeseries.append(df)
    #finally pop the ticker from the list
    tickers.pop()
    #save every 5 tickers in case it crashes
    if i%5==0:
        pickle.dump(timeseries, open(temp_timeseries_path,'wb'))
pickle.dump(timeseries, open(temp_timeseries_path,'wb'))    


# In[ ]:


#let's see which tickers caused errors.
#you can restart the loop with the tickers that had problems,
#(see above)


# In[ ]:


list(set(httperrors))


# In[ ]:


list(set(othererrors))


# In[ ]:


#load pickled timeseries
with open(temp_timeseries_path,'rb') as file:
    timeseries = pickle.load(file)


# In[ ]:


#make sure there are no duplicates
timeseries_unique = []
timeseries_tickers = []
for ts in timeseries:
    if ts.columns[0] not in timeseries_tickers:
        timeseries_tickers.append(ts.columns[0])
        timeseries_unique.append(ts)


# In[ ]:


#download SP500
req = head+'^GSPC'+tail
data = request.urlopen(req).read()
df = pd.read_csv(StringIO(data.decode()))
df = df[['timestamp','adjusted_close']]
df.timestamp = pd.to_datetime(df.timestamp, format='%Y-%m-%d')
df = df.set_index('timestamp')
df.columns = ['SP500']

timeseries_unique.append(df)


# In[ ]:


#download VIX
req = head+'^VIX'+tail
data = request.urlopen(req).read()
df = pd.read_csv(StringIO(data.decode()))
df = df[['timestamp','adjusted_close']]
df.timestamp = pd.to_datetime(df.timestamp, format='%Y-%m-%d')
df = df.set_index('timestamp')
df.columns = ['VIX']

timeseries_unique.append(df)


# In[ ]:


#let's join all the timeseries two at a time.
#we'll get one big dataframe with all timeseries.
jointwo = lambda a,b: a.join(b, how='outer')
from functools import reduce

timeseries_df = reduce(jointwo, timeseries_unique)


# In[ ]:


#rename columns
timeseries_names = [_.columns[0] for _ in timeseries_unique]


# In[ ]:


#apparently, there is some series that has
#minute-to-minute data for the day when the
#dataset is downloaded, this results in NaN's
#for all other daily timeseries.
timeseries_df['MMM'][-100:]


# In[ ]:


#Let's kill all the minute-by-minute datapoints.
#limit is the first timestamp of minute-by-minute.
#You will have to adjust it manually, because
#it's not worth the time to write code for doing it automatically.
limit = pd.to_datetime('2017-08-01-00:00:00')


# In[ ]:


#we'll keep all timestamps before it.
keep = timeseries_df.index <= limit
timeseries_df = timeseries_df.loc[keep,:]

#if you want to check that it worked
import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

plt.matshow(timeseries_df.iloc[:,:].notnull().values)
plt.show()
# In[ ]:


#finally, save the timeseries to a csv.
timeseries_df.to_csv(Paths.datadir.value + 'dataset.csv')


# ### Processing
# We're not allowed by the copyright terms to re-publish the data on the repository right away, but we can do so with a processed subset. We'll extract returns and reformat a bit the data.

# In[ ]:


timeseries = pd.read_csv(Paths.datadir.value + 'dataset.csv')


# In[ ]:


#let's keep the last nyears in the dataset
tradingyear = 252
annualize = np.sqrt(tradingyear)
nyears_keep = 5.5
timeseries = timeseries.iloc[-int(tradingyear*nyears_keep):,:]


# In[ ]:


#convert dates and set dataframe indices
timeseries.timestamp = pd.to_datetime(timeseries.timestamp, format = '%Y-%m-%d')
timeseries = timeseries.set_index('timestamp')
# move sp500 and vix to first columns, then all stocks sorted alphabetically
timeseries = timeseries[
        ['SP500', 'VIX']
        + list(sorted(
            [_ for _ in timeseries.columns.tolist() if (_!='SP500' and _!='VIX')]))
        ]
# let's kill the ~20% of stocks that didn't exist prior to
#earliest date picked in the dataset
#and the stocks with >1% missing data
#nevermind the biases introduced, they are besides the point
#of this analysis.
keep = (((timeseries.isnull().sum()/timeseries.shape[0])<.01)
       & timeseries.iloc[0,:].notnull().values)
timeseries = timeseries.loc[:,keep]
#there is a row of almost all missing points in this dataset.
#let's interpolate through it.
timeseries = timeseries.interpolate(method='linear')
#calculate linear returns, remove first row of prices to get same n rows
timeseries_returns = ((timeseries.shift(1) - timeseries)/timeseries).iloc[1:,:]
timeseries = timeseries.iloc[1:,:]
#slices for the dataframe:
indices_subset = slice(0,2)
stocks_subset = slice(2,None)


# In[ ]:


#we will now remove some of the prices and make a csv
#that is mostly nan. we'll keep the 100 days interval
#centered around the date when the CCAR scenario starts,
#because we need those prices for the calculation.
#this removal is done to comply with the license on the data
#that doesn't allow complete redistribution.


# In[ ]:


#load CCAR scenario
fed_sev_adverse = pd.read_csv(Paths.datadir.value + 'FedSeverelyAdverse.csv')
#convert dates and set dataframe indices
fed_sev_adverse.columns = [_.lower().replace(' ','_') for _ in fed_sev_adverse.columns]
fed_sev_adverse.date = pd.to_datetime(fed_sev_adverse.date, format = '%d-%m-%Y')
fed_sev_adverse = fed_sev_adverse.set_index('date')


# In[ ]:


#the interval of days to kill
kill1 = all_time = slice(None,
                 fed_sev_adverse.index[0]
                 - pd.Timedelta(50,'d'))
kill2 = all_time = slice(fed_sev_adverse.index[0]
                 + pd.Timedelta(50,'d'),
                 None)


# In[ ]:


#remove most of the prices from previous years,
#we only need the a few days
timeseries.loc[kill1,stocks_subset] = np.nan
timeseries.loc[kill2,stocks_subset] = np.nan


# In[ ]:


timeseries.to_csv(Paths.datadir_free.value + 'timeseries.csv')
timeseries_returns.to_csv(Paths.datadir_free.value + 'timeseries_returns.csv')

