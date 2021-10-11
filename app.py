#!/usr/bin/env python
# coding: utf-8

# <h3> Importing the Dependencies </h3>

# In[1]:


#Importing some libraries
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import math
from datetime import datetime
import datetime as DT

import matplotlib.pyplot as plt
import seaborn as sns

# For reading stock data from yahoo Finance (US / Global Region)
import yfinance as yf

#Importing Facebook Prophet modules
from prophet import Prophet

#Accuracy Metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

#Dash Specific Libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input,Output,State

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

from io import BytesIO
import base64

#from jupyter_dash import JupyterDash


# <h2> DASH FrontEnd Code </h2>

# In[3]:


#Dropdown & Ticker Data
ticker_data = pd.read_csv('symbols_valid_meta.csv')
options = []
for row_length in range(ticker_data.shape[0]):
    options.append({'label':'{}'.format(ticker_data.iloc[row_length,2]),'value':'{}'.format(ticker_data.iloc[row_length,1])})


# In[4]:


#Essential Stylings
logo = 'https://i.ibb.co/JjDGFqZ/Screenshot-2021-09-20-003243.png'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets = external_stylesheets,suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(children=[
        
        #Banner and Navigation Bar Section
        html.Div(children=[

            #Section for the Name of the Software on LHS
            html.Div(children=[html.H1(children='AI Based Stock Market Forecaster')],style={"margin-left":"1rem","color": "white","font-weight": "300",
            "font-family": "Open Sans","margin-bottom": "6rem"}),
            
            #Section for the Logo
            html.Img(style={"width": "200px", "height" : "62px"}, src=logo)],
            style={"display":"flex", "justifyContent": 'space-between', "height":'9.5vh', "marginBottom": '17px',
                   'background-color': '#141719',"border-left": "7px solid lightgray"}),

        #Division for two columns for the FrontEnd
        html.Div(children=[
            
            #Section for the LHS User Input Box
            html.Div(children=[
                html.Div(children=[
                html.Div(children=[html.H6(children="Company Information Input Section")],style={"font-weight": "900"}),html.Br(),
                html.P(children="Enter the Name of Company and Select the Timeframe to Predict & Analyze Stocks"),

                #Search Input for searching the Articles
                dcc.Dropdown(id='search-box', options = options,placeholder="Search Company Name",
                             style={"border": "none", "border-bottom": "2px solid gray","text-align": "center"}), html.Br(),

                #Section to search for the Medical Query
                html.Div(children=[

                #RadioButton to choose the origin of the Article
                html.Div(children=[

                dcc.RadioItems(id='Location-select',options=[{'label': '60 Days', 'value': '60'},{'label': '120 Days', 'value': '120'},{'label': '180 Days', 'value': '180'}],
                value='Global',labelStyle={'display': 'inline-block'})],style={"margin-bottom":"10px","margin-left":"3rem","margin-right":"3rem"}),
   
                #Search Button
                dbc.Button('Click to Analyze Stocks', id='submit-button', n_clicks=0, outline=True, color="secondary", className="mr-1",style={"box-shadow": "0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19)","border-radius": "12px"})],
                    
                style={"margin-top":"7px","margin-bottom":"45px"}),
                                   
                #Article Analysis Tab (Dynamically Fetching the Contents)
                html.Div(id="Dynamic_content_2",children=[]),
                
                #Section for choosing the website wise filtering (Dynamically Fetched)
                html.Div(id="Dynamic_content_1",children=[]),
                html.Br(),
                dcc.Loading(id="loading-2",type="default",fullscreen=False,children=html.Div(id="loading-output-2")),
                dcc.Loading(id="loading-1-A",type="default",fullscreen=False,children=html.Div(id="loading-output-1-A")),
                dcc.Loading(id="loading-1",type="default",fullscreen=False,children=html.Div(id="loading-output-1"))],style={"margin-left":"1rem","text-align": "center","margin-right":"1rem"})],

                #CSS Styling for the LHS Division
                style={"width":"27%","background": "rgba(255, 255, 255, 0.95)","border-style": "groove","height":'555px'}),

            #Inserting a blank column between two sections
            html.Div(children=[html.H4(children=" ")],style={"width":"1.5%"}),
            
            #Code for the RHS Division
            html.Div(children=[

              #Top Banner for the RHS Division with HTML Loading Element
              dcc.Loading(id="loading-3",type="default",fullscreen=False,children=html.Div(id="loading-output-3")),
                
              #Main HTML Division to create the footer Box
              html.Div(children=[
                  
                    #Section for the 1st row of the Footer
                    html.Div(children=[
                        html.Div(children=[html.H5(children='Stat1')],style={"text-align": "center","width":"25%","color": "white"}),
                        html.Div(children=[html.H5(children='Stat2')],style={"text-align": "center","width":"25%","color": "white"}),
                        html.Div(children=[html.H5(children='Stat3')],style={"text-align": "center","width":"25%","color": "white"}),
                        html.Div(children=[html.H5(children='Stat4')],style={"text-align": "center","width":"25%","color": "white"})],
                        style={"display":"flex"}),
                    
                    #Section for the 2nd row of the Footer
                    html.Div(id='Footer-div-A',children=[])],

                #Overall Styling of the Footer Section       
                style={"text-align": "center",'background-color': '#141719',
                        "height":'92px',"opacity":"0.9","margin-bottom":"2.3rem"}),
                

              html.Div(children=[
                html.Div(id='output-box-div-A',children=[]),
                
                #Main HTML Division to show the articles and the chart
                 html.Div(children=[  
                     
                    #Displaying the Charts in RHS of Output Box
                    html.Div(id='chart-d-1',children=[],style={"width":"470.5px","height":"270.5px"}),
                    
                    #Section for LHS Output box text
                    html.Div(children=[
                    
                    #Output for the Article Name
                    html.Div(id='Output-Article-name-div-A',children=[]),

                    #Output for the Horizontal Line
                    html.Div(id='Styling-element-div-A',children=[]),

                    #Output for the medical text
                    html.Div(id='Output-Article-NLP-Text-div-A',children=[]),html.Br(),

                    #Output for the url
                    html.Div(id='Output-Article-url-div-A',children=[])],
                    
                    #CSS Styling for the LHS Section (Articles)
                    style={"margin-left":"2rem","width":"470.5px","margin-right":"2rem",
                    "text-align": "justify","text-justify": "inter-word"})],

                    #Overall CSS Styling for the Article + Chart Section    
                    style={"display":"flex"})],
                    style={"background": "rgba(255, 255, 255, 0.95)",
                    "border-style": "groove","height":'440px',"marginBottom": '17px'})],
            
            #CSS Styling for the Right Division of the Page
            style={"width":"71.5%"})],

        #CSS design for both of the main LHS and RHS Division        
        style={"display": 'flex'}),

#Overall HTML Design
],style={"background-image": 'url("https://i.ibb.co/2MK309F/ibrahim-boran-a-Uw-E2-Dn-IPg-unsplash.jpg")'})


# <h3> Dash Callback Functions </h3>

# In[5]:


@app.callback(
[dash.dependencies.Output('loading-output-1-A','children'),
dash.dependencies.Output('output-box-div-A', 'children'),
dash.dependencies.Output('chart-d-1','children')],
dash.dependencies.Input('submit-button', 'n_clicks'),
[dash.dependencies.State('Location-select','value'),
dash.dependencies.State('search-box', 'value')])


def first_output(n_clicks,days,ticker):
    
    if n_clicks>0:
        
            #Getting the Company name 
            ticker_data = pd.read_csv('symbols_valid_meta.csv')

            #Obtaining the Company's Ticker
            for iter_ in range(ticker_data.shape[0]):
                if ticker == str(ticker_data.iloc[iter_,1]):
                    company = ticker_data.iloc[iter_,2]
                    break
        
            output_div_results = "Returning Stock Predictions for {}".format(company)
            output_div = html.Div(id='output-box-div',children=[html.Div(children=[html.H6(id='output-box',children=output_div_results)],style={"text-align": "center","height":"50px"})])
            
            #Getting the Stocks Data and doing the predictions using Prophet
            today = DT.date.today()
            
            #Converting the datatype
            days = int(days)

            #Creating initial dataframe
            apple_stocks_static = yf.download(ticker,'2015-01-01',datetime.now().strftime("%Y-%m-%d"))
            apple_stocks_static.reset_index(inplace=True)
            apple_stocks_static = apple_stocks_static[["Date","Close"]]
            apple_stocks_static =apple_stocks_static.rename(columns={"Date":"Date","Close":"Actual Values"})
            
            #Creating Static Dataframe
            Serial_no = []
            for serial in range(0,days):
                Serial_no.append(serial)
                
            dict = {'Serial_No':Serial_no}
            final_data = pd.DataFrame(dict)

            for split_condition in range(1,6):

                #On the basis of Split condition defining the starting date:
                if split_condition == 1:
                    start = today - DT.timedelta(days=(365*4)+days)
                    start = start.strftime("%Y-%m-%d")

                elif split_condition == 2:
                    start = today - DT.timedelta(days=(365*3)+days)
                    start = start.strftime("%Y-%m-%d")

                elif split_condition == 3:
                    start = today - DT.timedelta(days=912+days)
                    start = start.strftime("%Y-%m-%d")

                elif split_condition == 4:
                    start = today - DT.timedelta(days=730+days)
                    start = start.strftime("%Y-%m-%d")

                elif split_condition == 5:
                    start = today - DT.timedelta(days=365+days)
                    start = start.strftime("%Y-%m-%d")


                #Creating the dataset according to the condition
                apple_stocks = yf.download(ticker,start,datetime.now().strftime("%Y-%m-%d"))
                apple_stocks.reset_index(inplace=True)
                apple_stocks=apple_stocks[["Date","Close"]]
                apple_stocks=apple_stocks.rename(columns={"Date":"ds","Close":"y"})


                #Creating a training and testing data
                training_data = apple_stocks


                #Fitting the Facebook Prophet Model
                m = Prophet(daily_seasonality=True)
                m.fit(training_data) 
                future = m.make_future_dataframe(periods=days)
                forecast = m.predict(future)


                #Appending the Prediction column to the dataset
                prediction = forecast[['ds','yhat']]
                prediction = prediction.tail(days)
                final_data['Prediction{}'.format(split_condition)] = list(prediction['yhat'])
                final_data['Date'] = list(prediction['ds'])

            #Ensemble Model
            final_data['Stock Value Prediction'] = (final_data['Prediction1'] + final_data['Prediction2'] + final_data['Prediction3'] + final_data['Prediction4'] + final_data['Prediction5'])/5

            #Plotting the Data
            fig = px.line(final_data,x='Date',y='Stock Value Prediction')
            fig.update_layout(width=450, height=377, title="Stock Prediction for {} Days".format(days))
            fig = dcc.Graph(figure=fig)
            
            return['',output_div,fig]

if __name__ == '__main__':            
    app.run_server()


# In[ ]:




