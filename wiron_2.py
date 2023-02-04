# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 13:55:17 2022

@author: Komp
"""

import pandas as pd
import numpy as np
import sys
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates





sys.path.append('C:\\Users\\Public\\Documents\\IT\\Projects\\Schedule')
import schedule

raw_data=pd.read_excel('C:\\Users\\Public\\Documents\\IT\\Projects\\Wiron\\data\\bbg_datafeed.xlsx', parse_dates=['Dates'])

new_columns_names = {'WIRON Index':'WIRON',
                  'PZCFPLNI Index':'POLONIA',
                  'WIBRON Index':'WIBOR_ON',
                  'POREANN Index':'NBP_MAIN',
                  'WIBR1M Index':'WIBOR_1M',
                  'WIBR3M Index':'WIBOR_3M',
                  'WIRON3M Index':'WIRON_3M_BBG'
                  }

raw_data.rename(columns=new_columns_names, inplace=True)

dane_wiron       = pd.DataFrame(raw_data, columns=['Dates', 'WIRON']).dropna()
dane_polonia     = pd.DataFrame(raw_data, columns=['Dates.1', 'POLONIA']).dropna()
dane_wiboron     = pd.DataFrame(raw_data, columns=['Dates.2', 'WIBOR_ON']).dropna()
dane_nbpmain     = pd.DataFrame(raw_data, columns=['Dates.3', 'NBP_MAIN']).dropna()
dane_wibor1m     = pd.DataFrame(raw_data, columns=['Dates.4', 'WIBOR_1M']).dropna()
dane_wibor3m     = pd.DataFrame(raw_data, columns=['Dates.5', 'WIBOR_3M']).dropna()
dane_wiron3m_bbg = pd.DataFrame(raw_data, columns=['Dates.6', 'WIRON_3M_BBG']).dropna()

dane = dane_wiron.copy()
dane.rename(columns={'Dates':'value_date'}, inplace=True)

## dates calculations 
dane['fixing_date_wibor'] = dane['value_date'].apply(schedule.move_date_by_days, args=(-2, 'pln', 'pln'))        
dane['next_day'] = dane['value_date'].apply(schedule.move_date_by_days, args=(1,'pln', 'pln'))
dane['nb_of_days'] = (dane['next_day'] - dane['value_date']) / datetime.timedelta(days=1)
dane['compound_factor'] = dane['WIRON'] * dane['nb_of_days'] / 36500 + 1.0
dane['end_date_3m'] = pd.to_datetime(dane['value_date'].apply(schedule.mdbm_modified_following, args=(3, 'pln', 'pln')))
dane['end_date_1m'] = pd.to_datetime(dane['value_date'].apply(schedule.mdbm_modified_following, args=(1, 'pln', 'pln')))

## merging data on dates
temp_nbp = pd.DataFrame(pd.date_range(start=dane_nbpmain['Dates.3'][0], end=dane['value_date'].iloc[-1]), columns=['Dates.3'])
temp_nbp = pd.merge(temp_nbp, dane_nbpmain, how='left', on='Dates.3')
temp_nbp['NBP_MAIN'] = temp_nbp['NBP_MAIN'].ffill()

dane = pd.merge(dane, dane_polonia, left_on='fixing_date_wibor', right_on='Dates.1', how='left').drop(columns=['Dates.1'])
dane = pd.merge(dane, dane_wiboron, left_on='fixing_date_wibor', right_on='Dates.2', how='left').drop(columns=['Dates.2'])
dane = pd.merge(dane, temp_nbp, left_on='value_date', right_on='Dates.3', how='left').drop(columns=['Dates.3'])
dane = pd.merge(dane, dane_wibor1m, left_on='fixing_date_wibor', right_on='Dates.4', how='left').drop(columns=['Dates.4'])
dane = pd.merge(dane, dane_wibor3m, left_on='fixing_date_wibor', right_on='Dates.5', how='left').drop(columns=['Dates.5'])
dane = pd.merge(dane, dane_wiron3m_bbg, left_on='fixing_date_wibor', right_on='Dates.6', how='left').drop(columns=['Dates.6'])

# print(dane['POLONIA'].isnull().any())
# print(dane['WIBOR_ON'].isnull().any())
# print(dane['WIBOR_1M'].isnull().any())
# print(dane['WIBOR_3M'].isnull().any())

## compound wiron and spreads
cpd_1m_wiron = []
for i in range(dane.shape[0] - 22):
    d1 = dane['value_date'][i]
    d2 = dane['end_date_1m'][i]
    sub_df = dane.loc[(dane['value_date'] >= d1) & (dane['value_date'] < d2)]
    l = sub_df['nb_of_days'].sum()
    prod = sub_df['compound_factor'].product()
    r=(prod - 1.0) *36500 / l
    cpd_1m_wiron.append((dane['value_date'][i], r))

temp_cpd_1m_WIRON = pd.DataFrame(cpd_1m_wiron, columns=['Date', 'CPD_1M_WIRON'])    
dane = pd.merge(dane, temp_cpd_1m_WIRON, left_on='value_date', right_on='Date', how='left').drop(columns=['Date'])

cpd_3m_wiron = []
for i in range(dane.shape[0] - 65):
    d1 = dane['value_date'][i]
    d2 = dane['end_date_3m'][i]
    sub_df = dane.loc[(dane['value_date'] >= d1) & (dane['value_date'] < d2)]
    l = sub_df['nb_of_days'].sum()
    prod = sub_df['compound_factor'].product()
    r=(prod - 1.0) *36500 / l
    cpd_3m_wiron.append((dane['value_date'][i], r))
    
temp_cpd_3m_WIRON = pd.DataFrame(cpd_3m_wiron, columns=['Date', 'CPD_3M_WIRON'])    
dane = pd.merge(dane, temp_cpd_3m_WIRON, left_on='value_date', right_on='Date', how='left').drop(columns=['Date'])

dane['SPREAD_WIRON3M_BBG_WIBOR3M'] = dane['WIRON_3M_BBG'] - dane['WIBOR_3M']
dane['SPREAD_WIRON_NBP'] = dane['WIRON'] - dane['NBP_MAIN']
dane['SPREAD_CPD1MWIR_WIB1M'] = dane['CPD_1M_WIRON'] - dane['WIBOR_1M']
dane['SPREAD_CPD3MWIR_WIB3M'] = dane['CPD_3M_WIRON'] - dane['WIBOR_3M']

#dane.to_csv('dane.csv')
#dane.to_excel('dane.xlsx')


## calculation of means,  observation and last 180
mean_spread_wiron3m_wibor3m = [np.mean(dane['SPREAD_CPD3MWIR_WIB3M'])]*len(dane)
#print(mean_spread_wiron3m_wibor3m)
df_spread_wiron3m_wibor3m = dane[['value_date', 'nb_of_days', 'SPREAD_CPD3MWIR_WIB3M']].dropna()
#df_spread_wiron3m_wibor3m['reverse_cum_days'] = df_spread_wiron3m_wibor3m.loc[::-1, 'nb_of_days'].cumsum()
df_spread_wiron3m_wibor3m_last_180 = df_spread_wiron3m_wibor3m.tail(180)
mean_spread_wiron3m_wibor3m_last_180 = [np.mean(df_spread_wiron3m_wibor3m_last_180['SPREAD_CPD3MWIR_WIB3M'])]*180


## plotting

# fig, ax  = plt.subplots()
# ax.plot(dane['value_date'], dane['SPREAD_CPD3MWIR_WIB3M'], label = 'SPREAD_CPD3MWIR_WIB3M')
# ax.plot(dane['value_date'], mean_spread_wiron3m_wibor3m, label = 'MEAN ALL')
# ax.plot(df_spread_wiron3m_wibor3m_last_180['value_date'], mean_spread_wiron3m_wibor3m_last_180, label = 'MEAN LAST 180')
# ax.set_title('SPREAD CPD_3M_WIR vs WIBOR3M ')
# ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
# ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
# ax.set_ylabel('Spread %')
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax.set_xlabel('Time')
# ax.annotate('Mean total {:.2f}'.format(mean_spread_wiron3m_wibor3m[0]), xy=(datetime.date(2022, 8, 1),mean_spread_wiron3m_wibor3m[0]))
# ax.annotate('Mean last 180 {:.2f}'.format(mean_spread_wiron3m_wibor3m_last_180[0]), xy=(datetime.date(2022, 8, 1),mean_spread_wiron3m_wibor3m_last_180[0]))
# ax.grid()
# ax.legend()

# fig, ax2  = plt.subplots()
# ax2.plot(dane['value_date'],dane['SPREAD_WIRON3M_BBG_WIBOR3M'], label = 'SPREAD_WIRON3M_BBG_WIBOR3M')
# ax2.plot(dane['value_date'],dane['SPREAD_CPD3MWIR_WIB3M'], label = 'SPREAD_CPD3MWIR_WIB3M')
# ax2.set_title('SPREAD:\n WIRON3M_BBG vs WIBOR3M\n CPD_3M_WIR vs WIBOR3M ')
# ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
# ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
# ax2.set_ylabel('Spread %')
# ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax2.set_xlabel('Time')
# ax2.grid()
# ax2.legend()

# fig, ax3  = plt.subplots()
# ax3.plot(dane['value_date'],dane['WIRON'], label = "WIRON")
# ax3.plot(dane['value_date'],dane['WIBOR_ON'], label = "WIBOR_ON")
# ax3.plot(dane['value_date'],dane['POLONIA'], label = "POLONIA")
# ax3.plot(dane['value_date'],dane['NBP_MAIN'], label = "NBP_MAIN")
# ax3.set_title('WIRON, WIBOR ON, POLONIA')
# ax3.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.50))
# ax3.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
# ax3.set_ylabel('Rate %')
# ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax3.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax3.set_xlabel('Time')
# ax3.grid(which='both', color='b', linestyle='--', linewidth=0.5)
# ax3.legend()

# fig, ax4  = plt.subplots()
# ax4.plot(dane['value_date'],dane['WIBOR_1M'], label = "WIBOR_1M")
# ax4.plot(dane['value_date'],dane['WIBOR_3M'], label = "WIBOR_3M")
# ax4.plot(dane['value_date'],dane['CPD_1M_WIRON'], label = "CPD_1M_WIRON")
# ax4.plot(dane['value_date'],dane['CPD_3M_WIRON'], label = "CPD_3M_WIRON")
# ax4.set_title('COMPUND WIRON, WIBOR')
# ax4.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.50))
# ax4.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
# ax4.set_ylabel('Rate %')
# ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax4.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax4.set_xlabel('Time')
# ax4.grid()
# ax4.legend()

# fig, ax5  = plt.subplots()
# ax5.plot(dane['value_date'],dane['SPREAD_WIRON_NBP'], label = 'SPREAD_WIRON_NBP')
# ax5.plot(dane['value_date'],dane['SPREAD_CPD1MWIR_WIB1M'], label = 'SPREAD_CPD1MWIR_WIB1M')
# ax5.plot(dane['value_date'],dane['SPREAD_CPD3MWIR_WIB3M'], label = 'SPREAD_CPD3MWIR_WIB3M')
# ax5.set_title('SPREAD:\n WIRON vs NBP MAIN\n COMPUND WIRON vs WIBORs ')
# ax5.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
# ax5.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
# ax5.set_ylabel('Spread %')
# ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax5.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax5.set_xlabel('Time')
# ax5.grid()
# ax5.legend()