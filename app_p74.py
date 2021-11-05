


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from tensorflow.python.keras.backend import set_session
import tensorflow
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,Bidirectional,Conv2D, Activation,Dropout,Flatten,Dense,Conv2D,MaxPooling2D,ZeroPadding2D,BatchNormalization,ELU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.activations import elu
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

import snscrape.modules.twitter as snstwitter
import snscrape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from datetime import date, timedelta
import yfinance as yf

from nsetools import Nse
import plotly.express as px

import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from imblearn.over_sampling import SMOTEN,SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from pickle import dump
from pickle import load
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.feature_extraction.text import TfidfTransformer

from plotly.subplots import make_subplots
import en_core_web_sm
#Lemitization purpose 
nlp = en_core_web_sm.load()

#Lemmatization
wordnet=WordNetLemmatizer()

#Stop word
stop_words=stopwords.words('english')

#Lemitization purpose 
# nlp=spacy.load('en_core_web_sm')

# FUNCTION FOR STOCK DATA EXTRACTION
def stock_data_extractor(Ticker):
    Ticker_with_NS= Ticker+".NS"
    # pick the current date 
    current_date = date.today().isoformat() 
    # past 10 years data 
    past_10_years = (date.today()-timedelta(days=3650)).isoformat()
    # stock details extraction by YF
    data = yf.download(Ticker_with_NS, past_10_years,current_date) 
    
    return data

# FUNCTION FOR STOCK SUMMARY
def stock_summary(df,Ticker):
    try:
        # Nse is used to extract some last price
        Ticker_with_NS= Ticker+".NS"
        nse = Nse()
        # getting quote of the sbin
        quote = nse.get_quote(Ticker)
        
        # for PE ratio extraction
        yfdata = yf.Ticker(Ticker_with_NS)
    
        string_name = quote['companyName']
        st.header('*%s*' % string_name)
        st.write("NSE : ",quote['symbol']) 
        st.write(quote['lastPrice'],yfdata.info['currency'])
          
      
        a = float(quote['change'])
        if a > 0:
            st.write (quote['change'], '(',quote['pChange'],'%',')','\u2191','today',sep='') # '\u2191'up arrow
        elif a < 0:
            st.write (quote['change'], '(',quote['pChange'],'%',')','\u2193','today',sep='' ) # '\u2193'down arrow
        else:
            st.write (quote['change'], '(',quote['pChange'],'%',')','-','today',sep='' )
       
        
        # plotly Dynamic plot
        fig = px.line(df, x=df.index, y='Close', title='Market Summary')
        fig.update_xaxes(
            rangeslider_visible = True,
            rangeselector = dict(
                buttons = list([
                                dict(count = 1, label = '1m', step = 'month', stepmode = 'backward'),
                                dict(count = 2, label = '6m', step = 'month', stepmode = 'backward'),
                                dict(count = 3, label = '1yr', step = 'year', stepmode = 'backward'),
                                dict(count = 4, label = '3yr', step = 'year', stepmode = 'backward'),
                                dict(count = 5, label = '5yr', step = 'year', stepmode = 'backward'),
                                dict(count = 6, label = 'Max', step = 'year', stepmode = 'backward'),
                ])
            )
        )
        st.plotly_chart(fig)
  
    
        # Fundamental data 
        st.write("Open : ", yfdata.info["open"], "   Mkt Cap : ", yfdata.info["marketCap"], "   ROE : ", (int((yfdata.info["returnOnEquity"]*100)*100)/100),"%") 
        st.write("High : ", yfdata.info["dayHigh"], "  P/E Ratio : ", yfdata.info["trailingPE"], "    52-wk high : ", yfdata.info["fiftyTwoWeekHigh"])
        st.write("Low :  ", yfdata.info["dayLow"], "   Div Yield : ",(int((yfdata.info["dividendYield"]*100)*100)/100),"%","    52-wk low : ", yfdata.info["fiftyTwoWeekLow"])
        st.write("Recommendation by Yahoo Finance:", yfdata.info["recommendationKey"])
    except:
        pass

def date_fun(df):
    # create a date range from to today to 2021
    a=pd.date_range(start=date.today().isoformat(), end='2023-12-31' ,freq='D')
    # remove the weekends 
    b=pd.bdate_range(start=a[0],end=a[-1])
    # additional holiday dates apart from weekends for 2021
    none_trading_dates= ['2021-11-05', '2021-11-19']
    #removing the additional holidays from the non weekend dates
    b = b[~(b.strftime('%Y-%m-%d').isin(none_trading_dates))]
    # add the new dates without weedends to index of data frame strftime is isued to convert the format to Y-M-D
    df.index=b[:len(df)].strftime('%Y-%m-%d')
    
    return df

# FUNCTION FOR FORECAST
def forecast(df, time_step,forecast_days):
    # only use the close price   
    df=df['Close']
    scale = StandardScaler()
    X_input = df[-time_step:]

    
    forecast_pred=[]
    i=0
    n=forecast_days
    while i<=n:
        
        X_input = np.array(X_input[-time_step:])
     
        # convert data to array and then reshaping it to (70,1) -> which will
        X_input1 = scale.fit_transform(X_input.reshape(-time_step,1))
        
        # the date is scaled down and reshaped in the format - [samples, time steps, features]
        #(764, 70, 1) which is required by the model (did for X_train and test)
        X_input1 = np.reshape(X_input1,(1,time_step,1))
        
        # load the model
        #model= keras.models.load_model("Model_deep_3%.h5")
        model= keras.models.load_model("Model_deep_3%_new.h5")
        # Predict 
        pred = model.predict(X_input1)

        # Re scaling the output 
        pred_value = scale.inverse_transform(pred)

        # updating X_input with the last predicted value
        X_input=[*X_input, *pred_value]

        forecast_pred.append (*pred_value[-1])
    
        # data frame for predicted forecast
        forecast_df= pd.DataFrame(forecast_pred)
        
        forecast_df= date_fun(forecast_df)
        
        i=i+1
    
    return forecast_df


def Tweet_extraction(hashtag,days):
    
    tweets_list2=[]
    # how to add past 3 days on on your today date
    current_date = date.today().isoformat()   
    # getting the previous dates which will be used in until for extracting tweets 
    days_before = (date.today()-timedelta(days=days)).isoformat()

   
    for i, tweet in enumerate(snstwitter.TwitterSearchScraper('#'+ hashtag + ' since:'+days_before).get_items()):

            # Break @ desired number of tweets
            if i > 2000:
                break

            # Save the required details like content, date in list
            tweets_list2.append([tweet.date, tweet.content])

        # Creating a dataframe from the tweets list above
            tweets_df_new = pd.DataFrame(tweets_list2,
                                  columns=['Datetime', 'Text'])
    return tweets_df_new

# TWEET CLEANER
def tweet_cleaner (df):
    corpus = []
    for i in range(0, len(df)):

        # Removal of USer Tag eg - @shutter_con
        # Re.sub replace with regular expression
        tweet = re.sub("@[A-Za-z0-9]+"," ",df['Text'][i]) 

        # Removal of links 
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)

        # Removal of puntuations
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)

        # Converting Text to Lower case
        tweet = tweet.lower()

        # Spliting each words - eg ['I','was','happy']
        tweet = tweet.split()

        # Applying Lemmitization for the words eg: Argument -> Argue - Using Spacy Library
        tweet = nlp(' '.join(tweet))
        tweet = [token.lemma_ for token in tweet]

        # Removal of stop words
        tweet = [word for word in tweet if word not in stop_words]

        # Joining the words in sentences
        tweet = ' '.join(tweet)
        corpus.append(tweet)
        
    return corpus

# TWEET SENTIMENT
def Sentiment_plot (df):
    
    # tweet cleanewr function 
    cleaned_tweets=tweet_cleaner(df)
    
    # TFIDF -Pickel file
    loaded_TFIDF = load(open('Stock_tweet_sent.sav', 'rb'))
    
    # convert to number by TFIDF
    X=pd.DataFrame((loaded_TFIDF.transform(cleaned_tweets)).toarray())
    
    # SVC model loaded
    loaded_model= load(open('Stock_tweet_sentiment_SVC_model.sav','rb'))
    
    # Prediction
    pred =loaded_model.predict(X)
    
    # updating Senting against labels 
    pred_labels=["Positive" if i==1 else "Neutral" if i==0 else "Negative" for i in pred ]
    
    # count Plot for sentiment     
    fig=px.histogram(pred_labels,histnorm ='percent')
    st.plotly_chart(fig)



def main():
        # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:orange;padding:10px"> 
    <h1 style ="color:black;text-align:center;">Stock Analysis</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    ###########
    # sidebar #
    ###########
    
    # Slider for previous number of days tweet
    #tweets_days = st.sidebar.slider('Number of days for tweet', min_value=0, max_value=4)
    # Type stock Ticker 
    # CSV file loaded with Stock Ticker Details 
    stock_ticker_list=pd.read_csv("stocktickers.csv")
    # Select box will able to selct from the tickers list just by typing few intial letter
    st.set_option('deprecation.showfileUploaderEncoding', False)
    Ticker = st.sidebar.selectbox("Enter Stock Ticker",stock_ticker_list)


    
    # stock extraction function
    stock_data = stock_data_extractor(Ticker)
    # stock DF display
    #st.subheader("Stock Data")
    #st.write(stock_data)
    
        # Top gainers button
    nse=Nse()   
    if st.button('Top Gainers'):
        a =nse.get_top_gainers()
        gainers = pd.DataFrame(a)
        gainers= gainers[["symbol", "ltp"]]
        gainers.columns = ["Stock Ticker","Close Price"]
        st.write (gainers)
        
    # Top losers button
    if st.button('Top Losers') :
        b = nse.get_top_losers()
        losers = pd.DataFrame (b)
        losers= losers[["symbol", "ltp"]]
        losers.columns = ["Stock Ticker","Close Price"]
        st.write(losers)
    
    # Stock summary Plot
    stock_summay_plot=stock_summary(stock_data,Ticker)
    st.write(stock_summay_plot) 
    
    # Moving Avg
    st.subheader("Moving Average")
    stock_data['MA20'] = stock_data['Close'].rolling(window=20, min_periods=0).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(window=50, min_periods=0).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200, min_periods=0).mean()
    
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.10, subplot_titles=('Stock Price', 'Volume'), 
               row_width=[0.2, 0.7])

    fig3.add_trace(go.Candlestick(x=stock_data.index, open=stock_data["Open"], high=stock_data["High"],
                low=stock_data["Low"], close=stock_data["Close"], name="OHLC"), 
                row=1, col=1)
    fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data["MA20"], marker_color='grey',name="MA20"), row=1, col=1)
    fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data["MA50"], marker_color='skyblue',name="MA50"), row=1, col=1)
    fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data["MA200"], marker_color='violet',name="MA200"), row=1, col=1)

    fig3.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], marker_color='red', showlegend=False), row=2, col=1)

    fig3.update_layout(
    xaxis_tickfont_size=12,
    yaxis=dict(
        title='Price (INR/Share)',
        titlefont_size=14,
        tickfont_size=12,
        ),
    autosize=False,
    width=800,
    height=500,
    margin=dict(l=50, r=50, b=100, t=100, pad=4),
    
        )

    st.plotly_chart(fig3)
    
    
    # STOCK FORECAST FUNCTION
    forecast_data = forecast(stock_data, 70, 14)
    
    # Plot for Forecast 
    st.subheader("Forecast Graph")
    fig1=px.line(stock_data, x=stock_data.index, y="Close")
    fig1.add_scatter(x=forecast_data.index,y=forecast_data[0],mode='lines')
    
    fig1.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
                buttons = list([

                                dict(count = 3, label = '1yr', step = 'year', stepmode = 'backward'),

                ])
            )
        )
    
    st.plotly_chart(fig1)
     
    
    # Forecast DF display
    st.subheader("Predictions")
    
    st.write(forecast_data)


    # Tweet Extraction 
    tweet_data = Tweet_extraction(Ticker, 4)
    # Tweet DF display
    #st.subheader("Tweet Data")
    #st.write(tweet_data)
    
    #TWEET SENTIMENT ANALYSIS
    st.subheader("Twitter Sentiment")
    Sentiment_plot(tweet_data)
    # Forecast DF display
    #st.subheader("Forecast Prediction")
    #st.write(forecast_data)
    
    
    
 
    
  


    
    
    
if __name__ == '__main__':
    main()

