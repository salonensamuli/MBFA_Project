# required packages

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import streamlit as st
from yahooquery import Ticker

# streamlit

st.title('Undervalued Stocks Finder')
expander = st.expander('About')
expander.markdown('espander test')
st.header('header test')
st.subheader('subheader test')
st.caption('caption test')

# function to import yahoo data using the unofficial api

def get_stock_data(tickers):
    data = Ticker(tickers).get_modules('assetProfile price summaryDetail defaultKeyStatistics') 
    # assetProfile : industry , price : longName & regularMarketPrice , summaryDetail : twoHundredDayAverage , 
    # defaultKeyStatistics : forwardPE , profitMargins , beta , priceToBook , forwardEps , pegRatio , enterpriseToRevenue , enterpriseToEbitda
    return data

# scraping some index tickers

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
sp500 = sp500.to_list()

stoxx600 = pd.DataFrame(columns=['Ticker', 'Exchange'])

for i in range(1, 20):
    url = 'https://www.dividendmax.com/market-index-constituents/stoxx600.html/?page=' + str(i)
    data = pd.read_html(url)[0][['Ticker', 'Exchange']]
    stoxx600 = pd.concat([stoxx600, data])

for i in range(0, len(stoxx600)):
    if any(x in stoxx600.iloc[i]['Exchange'] for x in ['Frankfurt', 'Xetra', 'Berlin', 'Luxembourg']):
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.DE'
    elif 'London' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.L'
    elif 'Italian' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.MI'
    elif 'Amsterdam' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.AS'
    elif 'Stockholm' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.ST'
    elif 'Swiss' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.SW'
    elif 'Paris' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.PA'
    elif 'Brussels' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.BR'
    elif any(x in stoxx600.iloc[i]['Exchange'] for x in ['Madrid', 'Valencia', 'Barcelona']):
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.MC'
    elif 'Irish' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.L'
    elif 'Oslo' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.OL'
    elif 'Copenhagen' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.CO'
    elif 'Vienna' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.VI'
    elif 'Lisbon' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.LS'
    elif 'Warsaw' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.WA'
    elif 'Athens' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.AT'
    elif 'Helsinki' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.HE'

stoxx600 = stoxx600['Ticker'].to_list()

stoxx600 = [x.replace('..', '.') for x in stoxx600]
stoxx600 = [x.replace(' ', '-') for x in stoxx600]
stoxx600 = [x.replace('.', '-', 1) if x.count('.')==2 else x for x in stoxx600]

both = stoxx600 + sp500

# to make a streamlit dropdown to select  EU or US or both

# ask for wanted tickers

list_of_universes = ['sp500', 'stoxx600', 'both']
universe_choice = st.multiselect('choose your investment universe', list_of_universes)

dict_full = get_stock_data(universe_choice)

# deleting stocks that have no data 
# sidenote: we are losing some tickers because the STOXX600 website is a bit old, some stocks have changed exchanges etc.

delete = []
for i in dict_full.keys():
    if type(dict_full[i]) != dict:
        delete.append(str(i))
for i in delete:
    del dict_full[i]

# deleting the values from summaryDetail dictionary to avoid multiple same named columns in the df

delete = ['currency', 'forwardPE', 'beta']
for key in dict_full.keys():
    for i in delete:
        try:
            del dict_full[key]['summaryDetail'][i]
        except KeyError:
            pass

# ask what to analyze (DO AS STREAMLIT LIST)

# forwardPE , profitMargins , beta , priceToBook , forwardEps , pegRatio , enterpriseToRevenue , enterpriseToEbitda

list_of_multiples = ['forwardPE', 'profitMargins', 'beta', 'priceToBook', 'forwardEps', 'pegRatio', 'enterpriseToRevenue', 'enterpriseToEbitda']
multiple_choice = st.multiselect('choose your valuation multiple', list_of_multiples)

df = pd.DataFrame.from_dict({(i,j): dict_full[i][j] 
                           for i in dict_full.keys() 
                           for j in dict_full[i].keys()},
                        orient='index')
df = df[['longName', 'industry', 'currency', 'regularMarketPrice', 'twoHundredDayAverage', multiple_choice, ]]
df.reset_index(level=1, drop=True, inplace=True)
df = df.stack().unstack()
df.reset_index(inplace=True, names='ticker')
df[multiple_choice] = pd.to_numeric(df[multiple_choice])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=[multiple_choice], how='all', inplace=True)
df.drop(df.index[df[multiple_choice] < 0], inplace=True)

# because many London stocks are denoted in 0.01 pounds, we need to do some manipulation

# for i in range(0, len(df)):
#     if (df.iloc[i]['currency']=='GBp' and 0 < df.iloc[i]['forwardPE'] < 1):
#         df.iloc[i, df.columns.get_loc('forwardPE')] = df.iloc[i, df.columns.get_loc('forwardPE')] * 100
#         df.iloc[i, df.columns.get_loc('regularMarketPrice')] = df.iloc[i, df.columns.get_loc('regularMarketPrice')] / 100
#         df.iloc[i, df.columns.get_loc('twoHundredDayAverage')] = df.iloc[i, df.columns.get_loc('twoHundredDayAverage')] / 100

df.drop('currency', axis=1, inplace=True)

# checking for outliers

def is_outlier(s):
    lower_limit = 0
    upper_limit = s.mean() + (s.std() * 2)
    return ~s.between(lower_limit, upper_limit)

df = df[~df.groupby('industry', group_keys=False)[multiple_choice].apply(is_outlier)]

# creating industry rankings

df['industry_' + multiple_choice] = df[multiple_choice].groupby(df['industry']).transform('mean')
df['disc_' + multiple_choice] = df[multiple_choice] / df['industry_' + multiple_choice] - 1

# creating dataframe for "undervalued" stocks

df_uv = df.loc[df.groupby('industry')['disc_' + multiple_choice].idxmin()]
df_uv.sort_values(by='disc_' + multiple_choice, ascending=True, inplace=True)

print(df_uv)

# add a plot where the lowest valued is compared to the average of the industry

if st.button('Go!'):
    st.subheader('test results')
    st.dataframe(df_uv, height=3000)
