from datetime import date
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import quandl
import yfinance as yf



# Download historical data from Quandl
df_sensex = quandl.get('BSE/SENSEX', api_key='vKHe3RFS52fCx-4csoxH').reset_index()

# Rename columns to match Bitcoin code
df_sensex = df_sensex.rename(columns={'Date': 'Date', 'Close': 'Value'})

# Convert dates to datetime object for easy use
df_sensex['Date'] = pd.to_datetime(df_sensex['Date'])

# Sort data by date, just in case
df_sensex.sort_values(by='Date', inplace=True)

# Only include data points with existing price
df_sensex = df_sensex[df_sensex['Value'] > 0]

# Get the last price against USD
# Get the last price against USD
sensex_data = yf.download(tickers='^BSESN', period='1d', interval='1m')  # Ticker symbol for Sensex is ^BSESN
df_sensex.loc[df_sensex.index[-1] + 1] = [date.today(), sensex_data['Close'] = sensex_data['Close'].iloc[:, 0]]

# Append the latest price data to the dataframe
df_sensex.loc[df_sensex.index[-1]+1] = [date.today(), btcdata['Close'].iloc[-1]]
df_sensex['Date'] = pd.to_datetime(df_sensex['Date'])

# Calculate the `Risk Metric`
df_sensex['MA'] = df_sensex['Value'].rolling(window=252, min_periods=1).mean().dropna()
df_sensex['Preavg'] = (np.log(df_sensex['Value']) - np.log(df_sensex['MA'])) * df_sensex.index**.395

# Normalization to 0-1 range
df_sensex['avg'] = (df_sensex['Preavg'] - df_sensex['Preavg'].cummin()) / (df_sensex['Preavg'].cummax() - df_sensex['Preavg'].cummin())

# Predicting the price according to risk level
price_per_risk_sensex = {
    round(risk, 1): round(np.exp(
        (risk * (df_sensex['Preavg'].cummax().iloc[-1] - (cummin := df_sensex['Preavg'].cummin().iloc[-1])) + cummin) / df_sensex.index[-1]**.395 + np.log(df_sensex['MA'].iloc[-1])
    ))
    for risk in np.arange(0.0, 1.0, 0.1)
}

# Exclude the first 1000 days from the dataframe, because it's pure chaos
df_sensex = df_sensex[df_sensex.index > 1000]

# Title for the plots
AnnotationText = f"Updated: {sensex_data.index[-1]} | Price: {round(df_sensex['Value'].iloc[-1])} | Risk: {round(df_sensex['avg'].iloc[-1], 2)}"

# Plot BTC-USD and Risk on a logarithmic chart
fig = make_subplots(specs=[[{'secondary_y': True}]])

# Add BTC-USD and Risk data to the figure
fig.add_trace(go.Scatter(x=df_sensex['Date'], y=df_sensex['Value'], name='Price', line=dict(color='gold')))
fig.add_trace(go.Scatter(x=df_sensex['Date'], y=df_sensex['avg'], name='Risk', line=dict(color='white')), secondary_y=True)

# Add green (`accumulation` or `buy`) rectangles to the figure
opacity = 0.2
for i in range(5, 0, -1):
    opacity += 0.05
    fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0, fillcolor='green', opacity=opacity, secondary_y=True)

# Add red (`distribution` or `sell`) rectangles to the figure
opacity = 0.2
for i in range(6, 10):
    opacity += 0.1
    fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0, fillcolor='red', opacity=opacity, secondary_y=True)

fig.update_xaxes(title='Date')
fig.update_yaxes(title='Price (Sensex)', type='log', showgrid=False)
fig.update_yaxes(title='Risk', type='linear', secondary_y=True, showgrid=True, tick0=0.0, dtick=0.1, range=[0, 1])
fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
fig.show()

# Plot BTC-USD colored according to Risk values on a logarithmic chart
fig = px.scatter(df_sensex, x='Date', y='Value', color='avg', color_continuous_scale='jet')
fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
fig.show()

# Plot Predicting BTC price according to specific risk
fig = go.Figure(data=[go.Table(
    header=dict(values=['Risk', 'Price'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[list(price_per_risk_sensex.keys()), list(price_per_risk_sensex.values())],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])
fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk (Sensex)', 'y': 0.9, 'x': 0.5})
