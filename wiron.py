# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:32:16 2022

@author: Komp
"""

import datetime
import pandas as pd
import numpy as np
import random
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



sys.path.append('C:\\Users\\Public\\Documents\\IT\\Projects\\Schedule')
import schedule

holidays = pd.read_csv('..\\holidays\\pln.csv')
pln_holidays = holidays['pln']

#data reading
dane=pd.read_csv('data\\WIRON_dane_historyczne.csv', sep=';', parse_dates=['DATE'], dayfirst=True)
#dane.drop('INDEX', inplace=True, axis=1)
dane.rename(columns={'DATE':'value_date'}, inplace=True)


dane_nbpmain = pd.read_csv('data\\nbp_mainrate_20190101_20221127.csv', sep=';', parse_dates=['Date'], dayfirst=True)
dane_polonia = pd.read_csv('data\\polonia.csv', sep=';', parse_dates=['Date'], dayfirst=True)
dane_wiboron = pd.read_csv('data\\wibor_on.csv', sep=';', parse_dates=['Date'], dayfirst=True)
dane_wibor1m = pd.read_csv('data\\wibor_1m.csv', sep=';', parse_dates=['Date'], dayfirst=True)
dane_wibor3m = pd.read_csv('data\\wibor_3m.csv', sep=';', parse_dates=['Date'], dayfirst=True)

dane_wiron_bbg = pd.read_csv('data\\wiron_bbg.csv', sep=';', parse_dates=['Date'], dayfirst=True)
dane_wiron3m_bbg = pd.read_csv('data\\wiron3m_bbg.csv', sep=';', parse_dates=['Date'], dayfirst=True)



# dates calulations 
dane['fixing_date_wibor'] = dane['value_date'].apply(schedule.move_date_by_days, args=(-2, 'pln', 'pln'))        
dane['next_day'] = dane['value_date'].apply(schedule.move_date_by_days, args=(1,'pln', 'pln'))
dane['nb_of_days'] = (dane['next_day'] - dane['value_date']) / datetime.timedelta(days=1)
dane['compund_factor'] = dane['WIRON'] * dane['nb_of_days'] / 36500 + 1.0
dane['end_date_3m'] = pd.to_datetime(dane['value_date'].apply(schedule.mdbm_modified_following, args=(3, 'pln', 'pln')))
dane['end_date_1m'] = pd.to_datetime(dane['value_date'].apply(schedule.mdbm_modified_following, args=(1, 'pln', 'pln')))

# merging data
dane = pd.merge(dane, dane_polonia, left_on='fixing_date_wibor', right_on='Date', how='left').drop(columns=['Date'])
dane = pd.merge(dane, dane_wiboron, left_on='fixing_date_wibor', right_on='Date', how='left').drop(columns=['Date'])
dane = pd.merge(dane, dane_wibor1m, left_on='fixing_date_wibor', right_on='Date', how='left').drop(columns=['Date'])
dane = pd.merge(dane, dane_wibor3m, left_on='fixing_date_wibor', right_on='Date', how='left').drop(columns=['Date'])
dane = pd.merge(dane, dane_wiron3m_bbg, left_on='fixing_date_wibor', right_on='Date', how='left').drop(columns=['Date'])

dane['SPREAD_WIRON3M_BBG_WIBOR3M'] = dane['WIRON_3M_BBG'] - dane['WIBOR_3M']

#print(dane['POLONIA'].isnull().any())
#print(dane['WIBOR_ON'].isnull().any())
#print(dane['WIBOR_1M'].isnull().any())
#print(dane['WIBOR_3M'].isnull().any())


# merging NBP main rate
temp_nbp = pd.DataFrame(pd.date_range(start=datetime.date(2018, 12, 5), end=datetime.date(2022, 11, 26)), columns=['Date'])
temp_nbp = pd.merge(temp_nbp, dane_nbpmain, how='left', on='Date')
temp_nbp['NBP_MAIN'] = temp_nbp['NBP_MAIN'].ffill()
dane = pd.merge(dane, temp_nbp, left_on='value_date', right_on='Date', how='left').drop(columns=['Date'])

dane['SPREAD_WIRON_NBP'] = dane['WIRON'] - dane['NBP_MAIN']


# x1 = dane['value_date']
# x2 = dane['end_date_3m']
# t1 = x1[0]
# t2 = x2[0]
# sub_dane = dane.loc[(dane['value_date'] >= t1) & (dane['value_date'] < t2)]
# l = sub_dane['nb_of_days'].sum()
# prod = sub_dane['compund_factor'].product()
# r = (prod - 1.0) * 36500 / l
# sub_dane.to_excel('sub_dane.xlsx')


cpd_1m_wiron = []
for i in range(dane.shape[0] - 22):
    d1 = dane['value_date'][i]
    d2 = dane['end_date_1m'][i]
    sub_df = dane.loc[(dane['value_date'] >= d1) & (dane['value_date'] < d2)]
    l = sub_df['nb_of_days'].sum()
    prod = sub_df['compund_factor'].product()
    r=(prod - 1.0) *36500 / l
    cpd_1m_wiron.append((dane['value_date'][i], r))

temp_cpd_1m_WIRON = pd.DataFrame(cpd_1m_wiron, columns=['Date', 'CPD_1M_WIRON'])    
dane = pd.merge(dane, temp_cpd_1m_WIRON, left_on='value_date', right_on='Date', how='left').drop(columns=['Date'])
dane['SPREAD_CPD1MWIR_WIB1M'] = dane['CPD_1M_WIRON'] - dane['WIBOR_1M']


cpd_3m_wiron = []
for i in range(dane.shape[0] - 65):
    d1 = dane['value_date'][i]
    d2 = dane['end_date_3m'][i]
    sub_df = dane.loc[(dane['value_date'] >= d1) & (dane['value_date'] < d2)]
    l = sub_df['nb_of_days'].sum()
    prod = sub_df['compund_factor'].product()
    r=(prod - 1.0) *36500 / l
    cpd_3m_wiron.append((dane['value_date'][i], r))
    
temp_cpd_3m_WIRON = pd.DataFrame(cpd_3m_wiron, columns=['Date', 'CPD_3M_WIRON'])    
dane = pd.merge(dane, temp_cpd_3m_WIRON, left_on='value_date', right_on='Date', how='left').drop(columns=['Date'])
dane['SPREAD_CPD3MWIR_WIB3M'] = dane['CPD_3M_WIRON'] - dane['WIBOR_3M']

#dane.to_csv('dane.csv')
#dane.to_excel('dane.xlsx')


# plots
# implicit plotting

# check if WIRON ON on GPW and Bloomberg the same
# fig, ax  = plt.subplots()
# ax.plot(dane['value_date'],dane['WIRON'], label = "WIRON")
# ax.plot(dane_wiron_bbg['Date'],dane_wiron_bbg['WIRON_BBG'], label = "WIRON_BBG")
# ax.set_title('WIRON, WIRON_BBG')
# ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.50))
# ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
# ax.set_ylabel('Rate %')
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax.set_xlabel('Time')
# ax.grid(which='both', color='b', linestyle='--', linewidth=0.5)
# ax.legend()


# check WIRON 3M my calculation vs GPW (BBG)
# fig, ax  = plt.subplots()
# ax.plot(dane['value_date'],dane['CPD_3M_WIRON'], label = "CPD_3M_WIRON")
# ax.plot(dane_wiron3m_bbg['Date'],dane_wiron3m_bbg['WIRON_3M_BBG'], label = "WIRON_3M_BBG")
# ax.set_title('CPD_3M_WIRON, WIRON_3M_BBG')
# ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.50))
# ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
# ax.set_ylabel('Rate %')
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax.set_xlabel('Time')
# ax.grid(which='both', color='b', linestyle='--', linewidth=0.5)
# ax.legend()

# fig, ax  = plt.subplots()
# ax.plot(dane['value_date'],dane['SPREAD_WIRON3M_BBG_WIBOR3M'], label = 'SPREAD_WIRON3M_BBG_WIBOR3M')
# ax.plot(dane['value_date'],dane['SPREAD_CPD3MWIR_WIB3M'], label = 'SPREAD_CPD3MWIR_WIB3M')
# ax.set_title('SPREAD:\n WIRON3M_BBG vs WIBOR3M\n CPD_3M_WIR vs WIBOR3M ')
# ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
# ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
# ax.set_ylabel('Spread %')
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax.set_xlabel('Time')
# ax.grid()
# ax.legend()


mean_spread_wiron3m_wibor3m = [np.mean(dane['SPREAD_CPD3MWIR_WIB3M'])]*len(dane)
#print(mean_spread_wiron3m_wibor3m)

df_spread_wiron3m_wibor3m = dane[['value_date', 'nb_of_days', 'SPREAD_CPD3MWIR_WIB3M']].dropna()
#df_spread_wiron3m_wibor3m['reverse_cum_days'] = df_spread_wiron3m_wibor3m.loc[::-1, 'nb_of_days'].cumsum()
df_spread_wiron3m_wibor3m_last_180 = df_spread_wiron3m_wibor3m.tail(180)
mean_spread_wiron3m_wibor3m_last_180 = [np.mean(df_spread_wiron3m_wibor3m_last_180['SPREAD_CPD3MWIR_WIB3M'])]*180


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

# fig, ax  = plt.subplots()
# ax.plot(dane['value_date'],dane['WIRON'], label = "WIRON")
# ax.plot(dane['value_date'],dane['WIBOR_ON'], label = "WIBOR_ON")
# ax.plot(dane['value_date'],dane['POLONIA'], label = "POLONIA")
# ax.plot(dane['value_date'],dane['NBP_MAIN'], label = "NBP_MAIN")
# ax.set_title('WIRON, WIBOR ON, POLONIA')
# ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.50))
# ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
# ax.set_ylabel('Rate %')
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax.set_xlabel('Time')
# ax.grid(which='both', color='b', linestyle='--', linewidth=0.5)
# ax.legend()

# fig, ax2  = plt.subplots()
# ax2.plot(dane['value_date'],dane['WIBOR_1M'], label = "WIBOR_1M")
# ax2.plot(dane['value_date'],dane['WIBOR_3M'], label = "WIBOR_3M")
# ax2.plot(dane['value_date'],dane['CPD_1M_WIRON'], label = "CPD_1M_WIRON")
# ax2.plot(dane['value_date'],dane['CPD_3M_WIRON'], label = "CPD_3M_WIRON")
# ax2.set_title('COMPUND WIRON, WIBOR')
# ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.50))
# ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
# ax2.set_ylabel('Rate %')
# ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax2.set_xlabel('Time')
# ax2.grid()
# ax2.legend()

# fig, ax3  = plt.subplots()
# ax3.plot(dane['value_date'],dane['SPREAD_WIRON_NBP'], label = 'SPREAD_WIRON_NBP')
# ax3.plot(dane['value_date'],dane['SPREAD_CPD1MWIR_WIB1M'], label = 'SPREAD_CPD1MWIR_WIB1M')
# ax3.plot(dane['value_date'],dane['SPREAD_CPD3MWIR_WIB3M'], label = 'SPREAD_CPD3MWIR_WIB3M')
# ax3.set_title('SPREAD:\n WIRON vs NBP MAIN\n COMPUND WIRON vs WIBORs ')
# ax3.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
# ax3.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
# ax3.set_ylabel('Spread %')
# ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax3.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# ax3.set_xlabel('Time')
# ax3.grid()
# ax3.legend()





# explicit plotting

# ## WIRON i NBP RATE
# line WIRON
# fig1 = plt.figure(1)
# plt.plot(dane['value_date'],dane['WIRON'], label = "WIRON")
# plt.plot(dane['value_date'],dane['WIBOR_ON'], label = "WIBOR_ON")
# plt.plot(dane['value_date'],dane['POLONIA'], label = "POLONIA")
# plt.plot(dane['value_date'],dane['NBP_MAIN'], label = "NBP_MAIN")
# plt.title('WIRON, WIBOR ON, POLONIA')
# plt.yticks(np.arange(0.0, 7.5, 0.5))
# ax=plt.gca()
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# plt.legend()
# plt.grid()
# fig1.show()


# fig2 = plt.figure(2)
# plt.plot(dane['value_date'],dane['CPD_1M_WIRON'], label = "CPD_1M_WIRON")
# plt.plot(dane['value_date'],dane['WIBOR_1M'], label = "WIBOR_1M")
# plt.plot(dane['value_date'],dane['CPD_3M_WIRON'], label = "CPD_3M_WIRON")
# plt.plot(dane['value_date'],dane['WIBOR_3M'], label = "WIBOR_3M")
# plt.title('WIRON, WIBOR ON, POLONIA')
# plt.legend()
# fig2.show()

# fig3 = plt.figure(3)
# plt.plot(dane['value_date'],dane['SPREAD_CPD3MWIR_WIB3M'], label = "SPREAD CPD WIRON3M vs WIBOR3M")
# plt.plot(dane['value_date'],dane['SPREAD_CPD1MWIR_WIB1M'], label = "SPREAD CPD WIRON1M vs WIBOR1M")
# plt.title('SPREAD CPD WIRON vs WIBOR')
# ax=plt.gca()
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# plt.grid()
# plt.legend()
# fig2.show()


  

