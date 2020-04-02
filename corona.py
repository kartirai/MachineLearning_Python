# -*- codate_strding: utf-8 -*-
"""
Created on Thu Mar 26 00:04:11 2020

@author: raiajay
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
train = pd.read_csv('C:/Users/raiajay/Desktop/kartikey/train.csv')
train['date_str'] = train['Date'].copy()
train['Date'] = pd.to_datetime(train['Date'])
train['date_str']

pop_dict={'Afghanistan': 38928346,'Albania': 2877797,'Algeria': 43851044,'Andorra': 77265,'Argentina': 45195774,'Australia': 25499884,'Austria': 9006398,'Azerbaijan': 10139177,'Bahrain': 1701575,'Bangladesh': 164689383,'Belgium': 11589623,
          'Bosnia and Herzegovina': 3280819,'Brazil': 212559417,'Bulgaria': 6948445,'Burkina Faso': 20903273,'Canada': 37742154,'Chile': 19116201,'China': 1439323776,'Colombia': 50882891,'Costa Rica': 5094118,'Croatia': 4105267,'Cuba': 11326616,
          'Cyprus': 1207359,'Denmark': 5792202,'Dominican Republic': 10847910,'Ecuador': 17643054,'Egypt': 102334404,'Finland': 5540720,'France': 65273511,'Gabon': 2225734,'Germany': 83783942,'Ghana': 31072940,
          'Greece': 10423054,'Guatemala': 17915568,'Guyana': 786552,'Hungary': 9660351,'Iceland': 341243,'India': 1380004385,'Indonesia': 273523615,'Iran': 83992949,'Iraq': 40222493,'Ireland': 4937786,'Israel': 8655535,'Italy': 60461826,
          'Jamaica': 2961167,'Japan': 126476461,'Kazakhstan': 18776707,'Korea, South': 51269185,'Lebanon': 6825445,'Lithuania': 2722289,'Luxembourg': 625978,'Malaysia': 32365999,'Martinique': 375265,'Mauritius': 1271768,'Mexico': 128932753,
          'Moldova': 4033963,'Montenegro': 628066,'Morocco': 36910560,'Netherlands': 17134872,'Nigeria': 206139589,'North Macedonia': 2083374,'Norway': 5421241,'Pakistan': 220892340,'Panama': 4314767,'Paraguay': 7132538,'Peru': 32971854,
          'Philippines': 109581078,'Poland': 37846611,'Portugal': 10196709,'Romania': 19237691,'Russia': 145934462,'San Marino': 33931,'Saudi Arabia': 34813871,'Serbia': 8737371,'Seychelles': 98347,'Singapore': 5850342,
          'Slovakia': 5459642,'Slovenia': 2078938,'Somalia': 15893222,'South Africa': 59308690,'Spain': 46754778,'Sri Lanka': 21413249,'Sudan': 43849260,'Suriname': 586632,'Sweden': 10099265,'Switzerland': 8654622,'Thailand': 69799978,
          'Tunisia': 11818619,'Turkey': 84339067,'US': 331002651,'Ukraine': 43733762,'United Arab Emirates': 9890402,'United Kingdom': 67886011,'Uruguay': 3473730,'Uzbekistan': 33469203,'Venezuela': 28435940,'Vietnam': 97338579}

tp = pd.DataFrame(columns=['Country','ConfirmedCases','Fatalities'])
for coun in train['Country/Region'].unique():
    inter_db=train[train['Country/Region']==coun].groupby(['Date']).agg({'ConfirmedCases':['sum'],'Fatalities':['sum']}).reset_index()
    inter_db.columns=[	'Date',	'ConfirmedCases',	'Fatalities']
    tp.loc[tp.shape[0]]=[coun,inter_db.ConfirmedCases.max(),inter_db.Fatalities.max()]
tp.sort_values(by=['Fatalities'],ascending=False)

total_df = train.groupby(['Date','date_str','Country/Region','ConfirmedCases','Fatalities']).sum()
total_df = total_df.reset_index()
fig = px.scatter_geo(total_df,locations='Country/Region',locationmode='country names',color='Fatalities',hover_name='Country/Region',projection='natural earth',animation_frame='date_str',title='total fatalities over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()

country_db=train[train['Country/Region']=='Iran'].copy()
shift_val1=1
col_tocheck='ConfirmedCases'
dft=country_db[col_tocheck].shift(shift_val1).to_frame('shifted').join(country_db[col_tocheck].to_frame('val')).dropna()
print('Coefficient value is : ',np.corrcoef(dft['shifted'],dft['val'])[0][1])