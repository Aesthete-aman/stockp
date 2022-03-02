#!/usr/bin/env python
# coding: utf-8

# <h3> Importing the Dependencies </h3>

# In[6]:


#Importing some libraries
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
#import numpy as np
import math
from datetime import datetime
import datetime as DT

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

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

from jupyter_dash import JupyterDash
import pyfolio as pf


# <h2> DASH FrontEnd Code </h2>

# In[2]:


#Dropdown & Ticker Data
ticker_data = pd.read_csv('symbols_valid_meta.csv')
options = []
for row_length in range(ticker_data.shape[0]):
    options.append({'label':'{}'.format(ticker_data.iloc[row_length,2]),'value':'{}'.format(ticker_data.iloc[row_length,1])})


# In[46]:


#Essential Stylings
logo = 'https://i.ibb.co/JjDGFqZ/Screenshot-2021-09-20-003243.png'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets = external_stylesheets,suppress_callback_exceptions=True)
server = app.server
app.title = 'Stock Analyzer'

app.layout = html.Div(children=[
        
        #Banner and Navigation Bar Section
        html.Div(children=[

            #Section for the Name of the Software on LHS
            html.Div(children=[html.H1(children='AI Based Stock Market Analyzer')],style={"margin-left":"1rem","color": "white","font-weight": "300",
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

                #Section to search 
                html.Div(children=[

                #RadioButton 
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
                dcc.Loading(id="loading-2",type="graph",fullscreen=True,children=html.Div(id="loading-output-2")),
                dcc.Loading(id="loading-1-A",type="graph",fullscreen=True,children=html.Div(id="loading-output-1-A")),
                dcc.Loading(id="loading-1",type="graph",fullscreen=True,children=html.Div(id="loading-output-1"))],style={"margin-left":"1rem","text-align": "center","margin-right":"1rem"})],

                #CSS Styling for the LHS Division
                style={"width":"27%","background": "rgba(255, 255, 255, 0.9)","border-style": "groove","height":'555px'}),

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
                        html.Div(children=[html.H5(children='Calmar Ratio')],style={"text-align": "center","width":"25%","color": "white"}),
                        html.Div(children=[html.H5(children='Sharpe Ratio')],style={"text-align": "center","width":"25%","color": "white"}),
                        html.Div(children=[html.H5(children='Annual Return')],style={"text-align": "center","width":"25%","color": "white"}),
                        html.Div(children=[html.H5(children='Value at Risk')],style={"text-align": "center","width":"25%","color": "white"})],
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
                    html.Div(id='chart-d-1',children=[],style={"width":"470.5px"}),
                    
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
                    style={"background": "rgba(255, 255, 255, 0.9)",
                    "border-style": "groove","height":'440px',"marginBottom": '17px'})],
            
            #CSS Styling for the Right Division of the Page
            style={"width":"71.5%"})],

        #CSS design for both of the main LHS and RHS Division        
        style={"display": 'flex'}),

#Overall HTML Design
],style={"background-image": 'url("https://i.ibb.co/2MK309F/ibrahim-boran-a-Uw-E2-Dn-IPg-unsplash.jpg")'})


# <h3> Dash Callback Functions </h3>

# In[47]:


#Callback Function 1

@app.callback(
[dash.dependencies.Output("loading-output-1-A","children"),
dash.dependencies.Output('output-box-div-A', 'children'),
dash.dependencies.Output('Footer-div-A','children'),
dash.dependencies.Output('Output-Article-name-div-A','children'),
dash.dependencies.Output('Output-Article-url-div-A','children'),
dash.dependencies.Output('Styling-element-div-A','children'),
dash.dependencies.Output('Output-Article-NLP-Text-div-A','children'),
dash.dependencies.Output('chart-d-1','children'),
dash.dependencies.Output('Dynamic_content_2','children')],
dash.dependencies.Input('submit-button', 'n_clicks'),
[dash.dependencies.State('Location-select','value'),
dash.dependencies.State('search-box', 'value')])


def first_output(n_clicks,days,ticker):
    
    if n_clicks>0:
        
            global company
            global apple_stocks_1
            global predicted_data_1
        
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
            
            #Creating Static Dataframe
            Serial_no = []
            for serial in range(0,days):
                Serial_no.append(serial)
                
            dict = {'Serial_No':Serial_no}
            final_data = pd.DataFrame(dict)

            for split_condition in range(0,1):

                #On the basis of Split condition defining the starting date:
                if split_condition == 0:
                    start = today - DT.timedelta(days=(365*4)+days)
                    start = start.strftime("%Y-%m-%d")


                #Creating the dataset according to the condition
                apple_stocks = yf.download(ticker,start,datetime.now().strftime("%Y-%m-%d")) 
                apple_stocks.reset_index(inplace=True)
                apple_stocks=apple_stocks[["Date","Close"]]
                apple_stocks=apple_stocks.rename(columns={"Date":"ds","Close":"y"})


                #Creating a training and testing data
                training_data = apple_stocks


                #Fitting the Facebook Prophet Model
                m = Prophet(uncertainty_samples=None,daily_seasonality=True)
                m.fit(training_data) 
                future = m.make_future_dataframe(periods=days, include_history=False)
                forecast = m.predict(future)


                #Appending the Prediction column to the dataset
                prediction = forecast[['ds','yhat']]
                prediction = prediction.tail(days)
                final_data['Prediction{}'.format(split_condition)] = list(prediction['yhat'])
                final_data['Date'] = list(prediction['ds'])

            #Ensemble Model
            final_data['Stock Value Prediction'] = final_data['Prediction0']

            #Plotting the Data
            fig = px.line(final_data,x='Date',y='Stock Value Prediction')
            fig.update_layout(width=450, height=377, title="Stock Prediction for {} Days".format(days))
            fig = dcc.Graph(figure=fig)
            
            #Some Variables
            stock_name = 'Statistical Analysis of {} Stock Values'.format(company)
            styling = '-'*93
            text='Statistical Ratios are unavailable for Single Stock Predictions. The line chart on the left predicts the Stock Prices of {} for a period of {} days from Today. \n\n Select Number of Days from the radio button option available in LHS control window to analyze stock prices for a greater/lesser tenure.'.format(company,days)
            
            #Building variables for returning the dynamic content
            output_div_results = "Returning Stock Predictions for {}".format(company)
            output_div = html.Div(id='output-box-div',children=[html.Div(children=[html.H6(id='output-box',children=output_div_results)],style={"text-align": "center","height":"50px"})])
            
            #Footer Div Results
            footer_div = html.Div(id='Footer-div',children=[html.Div(children=[
                        html.Div(children=[html.H5(id='votes',children='NA')],style={"text-align": "center","width":"25%"}),
                        html.Div(children=[html.H5(id='status',children='NA')],style={"text-align": "center","width":"25%"}),
                        html.Div(children=[html.H5(id='website',children='NA')],style={"text-align": "center","width":"25%"}),
                        html.Div(children=[html.H5(id='acceptance',children='NA')],style={"text-align": "center","width":"25%"})],
                        style={"display":"flex","color":"white"})])
            
            #File Downloading Options
            start1 = today - DT.timedelta(days=(365*4)+days)
            start1 = start1.strftime("%Y-%m-%d")
            apple_stocks_1 = yf.download(ticker,start1,datetime.now().strftime("%Y-%m-%d")) 
            predicted_data_1 = final_data
            
            
            down1 = html.Div(id='New',children=
                             [html.Div([html.Button("Download Stock Values", id="btn_csv"), dcc.Download(id="download-text-index")]),
                              html.Div(children=[html.H4(children=" ")],style={"width":"6px"}),
                              html.Div([html.Button("Download Predictions", id="btn_csv_1"), dcc.Download(id="download-text-index1")])],style={"display":"flex"})
            
            output_article_name_div = html.Div(id='Output-Article-name-div',children=[html.Div(children=[html.H6(children=stock_name, id="Output-Article-name")],style={"text-align":"center","font-weight": "bold","margin-bottom":"5px"})])
            url_div = html.Div(id='Output-Article-url-div',children=[html.Div(children=[down1],style={"text-align":"left","font-size": "0.9em","text-align": "justify","text-justify": "inter-word","display":"flex"})])
            styling_div = html.Div(id='Styling-element-div',children=[html.Div(children=styling,id="Styling-element")])            
            article_text_div = html.Div(id='Output-Article-NLP-Text-div',children=[html.Div(children=[html.Div(children=text, id="Output-Article-NLP-Text"),html.Br(),html.P('Click the Below Buttons to download the 4-year Actual or Predicted Values of {} Stocks'.format(company))],style={'overflow-y':"auto","overflow-x":"hidden","height":"200px","font-family": "Georgia"})])
            
            #For Line Chart
            word_c1 =html.Div(children=[html.Div(id='chart',children=fig)])
            
            d2 = html.Div(children=[
                html.Div(children=[html.H6(children="Pair Wise Stock Trading")],style={"font-weight": "900"}),
                html.P(children="Select the List of Companies to Analyze Pair Wise Stock Trading"),

                #Search Input for searching the Articles
                dcc.Dropdown(id='search-box-1', options = options,multi=True,clearable=False,
                             style={"border": "none", "border-bottom": "2px solid gray","text-align": "center"}), html.Br(),
   
                #Search Button
                dbc.Button('Analyze Pair Trading', id='submit-button-1', n_clicks=0, outline=True, color="secondary", className="mr-1",style={"box-shadow": "0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19)","border-radius": "12px"})],
                    
                style={"margin-top":"7px","margin-bottom":"45px"})
            

            #Returning all the parameters in a sequential order
            return ['',output_div,footer_div,output_article_name_div,url_div,styling_div,article_text_div,word_c1,d2]
        

#-------------------------------------------------------------------------------------------------------------------------
@app.callback(
[dash.dependencies.Output("download-text-index","data")],
[dash.dependencies.Input("btn_csv", "n_clicks")],
prevent_initial_call=True)  

def first_output(n_clicks):
    
    if n_clicks>0:
        
        return [dcc.send_data_frame(apple_stocks_1.to_csv, "{} Stocks.csv".format(company))]
    
    
    
@app.callback(
[dash.dependencies.Output("download-text-index1","data")],
[dash.dependencies.Input("btn_csv_1","n_clicks")],
prevent_initial_call=True)  

def first_output(n_clicks):
    
    if n_clicks>0:
        
        return [dcc.send_data_frame(predicted_data_1.to_csv, "{} Predictions.csv".format(company))]
    

#-------------------------------------------------------------------------------------------------------------------------
        
        
@app.callback(
[dash.dependencies.Output("loading-output-2","children"),
dash.dependencies.Output('output-box-div', 'children'),
dash.dependencies.Output('Footer-div','children'),
dash.dependencies.Output('Output-Article-name-div','children'),
dash.dependencies.Output('Output-Article-url-div','children'),
dash.dependencies.Output('Styling-element-div','children'),
dash.dependencies.Output('Output-Article-NLP-Text-div','children'),
dash.dependencies.Output('chart','children')],
dash.dependencies.Input('submit-button-1', 'n_clicks'),
[dash.dependencies.State('search-box-1', 'value')])


def first_output(n_clicks,ticker):
    
    global perf_stats_all
    global plot_data
    global company_values
    
    if n_clicks>0:
        
            tickers_list = ticker
            
            # Import pandas and create a placeholder for the data
            data = pd.DataFrame(columns=tickers_list)

            # Fetch the data
            for ticker in tickers_list:
                 data[ticker] = yf.download(ticker, period='5y',)['Adj Close']

            data = data.pct_change().dropna().mean(axis=1)
            plot_data = data
            
            # Get the full tear sheet
            from pyfolio import timeseries 
            perf_func = timeseries.perf_stats 
            perf_stats_all = perf_func(returns=data, positions=None, transactions=None, turnover_denom="AGB")
                
            company = []
            
            #Getting the Company name 
            ticker_data = pd.read_csv('symbols_valid_meta.csv')

            #Obtaining the Company's Ticker
            for i in range(len(tickers_list)):
                for iter_ in range(ticker_data.shape[0]):
                    if tickers_list[i] == str(ticker_data.iloc[iter_,1]):
                        company.append(ticker_data.iloc[iter_,2])
                        break
            
            company_values = company
            
            output_div_results = "Returning Stock Analyses and Ratios for Selected Companies"
            output_div = html.Div(id='output-box-div',children=[html.Div(children=[html.H6(id='output-box',children=output_div_results)],style={"text-align": "center","height":"50px"})])
            
            #Getting the Stocks Data and doing the predictions using Prophet
            today = DT.date.today()
            
            Serial_no = []
            company1 = []
            predictions = []
            date = []
            
            for serial in range(0,180*len(tickers_list)):
                Serial_no.append(serial)
                        
            for j in range(len(tickers_list)):
                company1 += [company[j]]*180

            dict = {'Serial_No':Serial_no,'Company':company1}
            final_data = pd.DataFrame(dict)
            
            for i in range(len(tickers_list)):
                
                        ticker = tickers_list[i]
    
                        for split_condition in range(1):

                            #On the basis of Split condition defining the starting date:
                            if split_condition == 0:
                                start = today - DT.timedelta(days=(365*4)+180)
                                start = start.strftime("%Y-%m-%d")


                            #Creating the dataset according to the condition
                            apple_stocks = yf.download(ticker,start,datetime.now().strftime("%Y-%m-%d")) 
                            apple_stocks.reset_index(inplace=True)
                            apple_stocks=apple_stocks[["Date","Close"]]
                            apple_stocks=apple_stocks.rename(columns={"Date":"ds","Close":"y"})


                            #Creating a training and testing data
                            training_data = apple_stocks


                            #Fitting the Facebook Prophet Model
                            m = Prophet(uncertainty_samples=None,daily_seasonality=True)
                            m.fit(training_data) 
                            future = m.make_future_dataframe(periods=180, include_history=False)
                            forecast = m.predict(future)


                            #Appending the Prediction column to the dataset
                            prediction = forecast[['ds','yhat']]
                            prediction = prediction.tail(180)
                            
                            predictions += list(prediction['yhat'])
                            date += list(prediction['ds'])
            
            
            final_data['Stock Value Prediction'] = predictions
            final_data['Date'] = date

                            
            #Plotting the Data
            fig = px.line(final_data,x='Date',y='Stock Value Prediction',color='Company')
                        
            fig.update_layout(width=450, height=377) 
            fig.update_layout(showlegend=False,title="Pair wise Stock Analysis")
            fig = dcc.Graph(figure=fig)
            
            
            #Some Variables
            stock_name = 'Statistical Analysis of Selected Stock Values'
            styling = '-'*93
            text1 = 'Calmar Ratio ({}) denotes performance of investment funds. The higher the Calmar ratio, the better the invesment performed on a risk-adjusted basis during a 36 month period.'.format(round(perf_stats_all[4],2))
            text2 = 'Sharpe Ratio ({}) describes expected returns vs. Risks assuming normal distributions. A good Sharpe ratio (Above 1) indicates a high degree of expected return for a relatively low amount of risk made in an invesment'.format(round(perf_stats_all[3],2))
            
            
            #Footer Div Results
            footer_div = html.Div(id='Footer-div',children=[html.Div(children=[
                        html.Div(children=[html.H5(id='votes',children=round(perf_stats_all[4],2))],style={"text-align": "center","width":"25%"}),
                        html.Div(children=[html.H5(id='status',children=round(perf_stats_all[3],2))],style={"text-align": "center","width":"25%"}),
                        html.Div(children=[html.H5(id='website',children=round(perf_stats_all[0],2))],style={"text-align": "center","width":"25%"}),
                        html.Div(children=[html.H5(id='acceptance',children=round(perf_stats_all[-1],2))],style={"text-align": "center","width":"25%"})],
                        style={"display":"flex","color":"white"})])
            
            down2 = html.Div(id='New1',children=
                             [html.Div([html.Button("Download Dashboard", id="btn_csv10"), dcc.Download(id="download-text-index12")]),
                              html.Div(children=[html.H4(children=" ")],style={"width":"6px"}),
                              html.Div([html.Button("Download Statistics", id="btn_csv_11"), dcc.Download(id="download-text-index13")])],style={"display":"flex"})
            
            
            output_article_name_div = html.Div(id='Output-Article-name-div',children=[html.Div(children=[html.H6(children=stock_name, id="Output-Article-name")],style={"text-align":"center","font-weight": "bold","margin-bottom":"5px"})])
            url_div = html.Div(id='Output-Article-url-div',children=[html.Div(children=[down2],style={"text-align":"left","font-size": "0.9em","text-align": "justify","text-justify": "inter-word"})])
            styling_div = html.Div(id='Styling-element-div',children=[html.Div(children=styling,id="Styling-element")])            
            article_text_div = html.Div(id='Output-Article-NLP-Text-div',children=[html.Div(children=[html.Div(children=text1, id="Output-Article-NLP-Text"),
                                                                                            html.Br(),
                                                                                            html.P(text2)],style={'overflow-y':"auto","overflow-x":"hidden","height":"200px","font-family": "Georgia"})])
            
            #For Line Chart
            #word_c1 =html.Div(children=[html.Div(id='chart',children=fig)])
            
            d2 = html.Div(children=[
                html.Div(children=[html.H6(children="Pair Wise Stock Trading")],style={"font-weight": "900"}),
                html.P(children="Select the List of Companies to Analyze Pair Wise Stock Trading"),

                #Search Input for searching the Articles
                dcc.Dropdown(id='search-box-1', options = options,multi=True,clearable=True,
                             style={"border": "none", "border-bottom": "2px solid gray","text-align": "center"}), html.Br(),
   
                #Search Button
                dbc.Button('Click to Analyze Stocks', id='submit-button-1', n_clicks=0, outline=True, color="secondary", className="mr-1",style={"box-shadow": "0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19)","border-radius": "12px"})],
                    
                style={"margin-top":"7px","margin-bottom":"45px"})
            

            #Returning all the parameters in a sequential order
            return ['',output_div,footer_div,output_article_name_div,url_div,styling_div,article_text_div,fig]

#-------------------------------------------------------------------

@app.callback(
[dash.dependencies.Output("download-text-index12","data")],
[dash.dependencies.Input("btn_csv10", "n_clicks")],
prevent_initial_call=True)  

def first_output(n_clicks):
    
    if n_clicks>0:
        
        bt_returns = plot_data
        fig = plt.figure(1)
        plt.subplot(3,3,1)
        pf.plot_annual_returns(bt_returns)
        plt.subplot(3,3,2)
        pf.plot_monthly_returns_dist(bt_returns)
        plt.subplot(3,3,3)
        pf.plot_monthly_returns_heatmap(bt_returns)


        plt.subplot(3,3,4)
        pf.plot_drawdown_underwater(bt_returns)


        plt.subplot(3,3,5)
        pf.plot_return_quantiles(bt_returns)

        plt.subplot(3,3,6)
        pf.plot_rolling_returns(bt_returns)
        plt.subplot(3,3,7)
        pf.plot_rolling_sharpe(bt_returns)
        
        fig.suptitle('Plots for Pair Wise Stock Analysis for {}'.format(company_values), fontsize=16)


        #plt.tight_layout()
        fig.set_size_inches(19.5, 18.5)

        pp = PdfPages('Pair_Trade_Dashboard.pdf')
        pp.savefig(fig)
        pp.close()
        
        
        return [dcc.send_file("./Pair_Trade_Dashboard.pdf")]
    
    
    
@app.callback(
[dash.dependencies.Output("download-text-index13","data")],
[dash.dependencies.Input("btn_csv_11","n_clicks")],
prevent_initial_call=True)  

def first_output(n_clicks):
    
    if n_clicks>0:
        
        s = pd.Series(perf_stats_all,name="Value")
        s = s.to_frame()
        return [dcc.send_data_frame(s.to_csv, "Pair-Trade Statistics.csv")]         
    
#------------------------------------------------------
#------------------------------------------------------

if __name__ == '__main__':
    app.run_server()
