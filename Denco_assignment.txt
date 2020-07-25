# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:54:17 2020

@author: kamakshi Gupta
"""
import pandas as pd

dencodata = pd.read_csv('denco.csv') #same folder
dencobackup=dencodata.copy()
dencodata.columns

#%%
# Who are the most loyal customers
# Make customer table, See customer transaction,
# Sort Customer Transaction,
# How many times are these customers buying from me
# Select the Top 5 or 10 rows (Sorted in Descending Order of Frequency)

dencodata['count']=1 #added another column for keeping count of transactions
dencodata.columns
custtable = dencodata[['custname','revenue','cost','margin','count']] 
loyalcust = custtable.groupby("custname").sum().sort_values("count", ascending=False)
#groupby unique customer name, taking sum of column value then sorting by count for getting number of transactions
print('Top 10 customers with max number of transactions in descending order:')
loyalcust.head(10)

#%%
# Which customers contribute the most to their revenue
# Sum the revenue by each customer
# Sort revenue by customers in descending Order

valuablecust = custtable.groupby("custname").sum().sort_values("revenue", ascending=False)
#sorting by revenue 
print('Top 10 customers with max revenue in descending order:')
valuablecust.head(10)

#%%
# What part numbers bring in to significant portion of revenue
# Sum/ Group the revenue by part no
# Sort the revenue by decreasing order
# Top revenue by part nos
parts = dencodata[['partnum','revenue','cost','margin','count']]
valueparts = parts.groupby("partnum").sum().sort_values("revenue", ascending=False)
#groupby part number sum of column values then sorting by revenue
print('Top 10 parts with max revenue in descending order:')
valueparts.head(10)

#%%
# What parts have the highest profit margin ?
# Sum the margin by partno
# Sort the margin by decreasing order
# Parts contributing highest profit margin

profitparts = parts.groupby("partnum").sum().sort_values("margin", ascending=False)
#soritng by margin
print('Top 10 parts with max margin in descending order:')
profitparts.head(10)

#%%
#Who are their top buying customers
# solution - loyal customers with max transactions
print("top buying customers by number of transactions: ")
loyalcust.head(10)
# Who are the customers who are bringing more revenue
print("top buying customers by revenue: ")
valuablecust.head(10)
