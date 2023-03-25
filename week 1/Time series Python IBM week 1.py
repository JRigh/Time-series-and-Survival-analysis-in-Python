#---------------------------------------------
# Time series and survival analysis IBM course
#---------------------------------------------

# imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from IPython.display import display
import os
os.chdir('data')
from colorsetup import colors, palette
sns.set_palette(palette)
# ignore warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.1f}'.format
plotsize = (13, 5)

# read data frame as an excel file
df = pd.read_excel("Sample - Superstore.xls")
df.columns

# data cleaning and reshaping
variables = ['Order Date', 'Category', 'Sales']
group_variables = variables[:2]
outcome_variable = variables[2]
base = df.groupby(group_variables)[outcome_variable].sum().reset_index()

base.head()

# pandas dataframe types
for x in base.columns:
    print(x, type(base[x]), base[x].dtype)
    
# working with numpy arrays
order_date = np.array(base['Order Date'])
category = np.array(base['Category'])
sales = np.array (base['Sales'])

print('Order Date', type(order_date), order_date.dtype)
print('Category', type(category), category.dtype)
print('Sales', type(sales), sales.dtype)

df_from_numpy = pd.DataFrame({'Order Date':order_date, 'Category':category, 'Sales':sales})

order_date_daily = np.array(order_date, dtype='datetime64[D]')
order_date_monthly = np.array(order_date, dtype='datetime64[M]')

# working with the pandas datetimeIndex

print(base.head())
print('\n Unique categories:')
print(base['Category'].unique())

base.set_index('Order Date', inplace=True)
# Note that without inplace=True, it will output the results without changing the data

base.head()

# subsetting data
# Observations in 2014
print(base['2011'].head())
print('\n')
# Observations in a range of dates, subset of columns:
print(base[base['Category'] == 'Office Supplies']['2011':'2012-02'].head())

# datetime components

#base.set_index('Order Date', inplace=True)
print('Day:', base.index.day, '\n')
print('Week:', base.index.week, '\n')
base['DayofWeek'] = base.index.dayofweek # Day of Week: Monday=0, Sunday=6
print(base.head())
# Note: use dt method when the date variable is not part of the index: 
# df['Order Date'].dt.dayofweek.head()
del(base['DayofWeek'])

# pandas pivot method

base.reset_index(inplace=True)
# Note if we didn't reset the index, we could use index=None below
sales_pivot = base.pivot(index='Order Date', columns='Category', values='Sales')
sales_pivot.head()

# long to wide (unstacking)
sales = base.set_index(['Order Date', 'Category']).unstack('Category').fillna(0)

# Note -- 2 levels of column names, the original variables are in columns.levels[0], 
# newly-created category variable names are in columns.levels[1]. This can be reset completely:
# sales.columns = sales.columns.levels[1].rename(None)
# Alternatively, keeping 'Sales' as a level 0 name allows us to refer to the variables jointly (sales['Sales'])
sales.columns = sales.columns.levels[1].rename(None)
sales.head()

print(sales.index)
print('\nUnique dates in our data: ', len(sales.index.unique()), 'Days')

# generating a complete index and setting frequency
print('\nUnique dates in our data: ', len(sales.index.unique()), 'Days')
our_date_range = sales.index.max() - sales.index.min()

# Calculate number of days in date range
print('Total days in our date range:', our_date_range.days, 'Days')
#date_range = pd.date_range(min(sales.index), max(sales.index))

new_index = pd.date_range(sales.index.min(), sales.index.max())
print(new_index)

sales_new = sales.reindex(new_index, fill_value=0)


# upsampling
sales_weekly = sales_new.resample('W').sum()
print('Weekly Sales')
print(sales_weekly.head(), '\n')

sales_monthly = sales_new.resample('M').sum()
print('Monthly Sales')
print(sales_monthly.head(), '\n')

sales_quarterly = sales_new.resample('Q').sum()
print('Quarterly Sales')
print(sales_quarterly.head(), '\n')

sales_annual = sales_new.resample('Y').sum()
print('Annual Sales')
print(sales_annual.head())

# downsampling
# Note that downsampling (from Annual to Monthly for example) produces missing values:
sales_monthly_from_annual = sales_annual.resample('M')
#print('Monthly from Annual Sales')
#sales_monthly_from_annual.interpolate(method='linear').head()
print(sales_monthly_from_annual.interpolate(method='spline', order=3).head())

# resampling by changing frequency directly
sales_daily = sales.asfreq('D')
sales_businessday = sales.asfreq('B')
sales_hourly = sales.asfreq('h')
# This will generate missing values:
sales_hourly.head()

# stationarity transformation
# Variable First Difference
print('Monthly Sales, First Difference \n', sales_monthly.diff().head())

# Variable Percent Change
print('\nMonthly Sales % Change \n', sales_monthly.pct_change().head())

# Log Sales
print('\nlog(1+Monthly Sales) \n', np.log(1 +  sales_monthly).head())

# Add % change to original data:
sales_monthly.join(sales_monthly.pct_change().add_suffix('_%_Change')).head()


# rolling averages
window_size = 7
rolling_window = sales_new.rolling(window_size)
print('Rolling Mean')
print(rolling_window.mean().dropna().head())
print('\nRolling St. Dev')
print(rolling_window.std().dropna().head())
print('\nCumulative Sales')
print(sales_new.cumsum().dropna().head())

# visualization
sales_quarterly.plot(figsize=plotsize, title='Quarterly Sales')
#plt.title('Monthly Sales')
sales_monthly.plot(figsize=plotsize, title='Monthly Sales')
#plt.title('Monthly Sales')
sales_weekly.plot(figsize=plotsize, title='Weekly Sales')
#plt.title('Monthly Sales')

# rolling averages and cumulative sales
#rolling_window.mean().plot(figsize=plotsize, title='Daily Sales, 7-day Rolling Average')
rolling_window.std().plot(figsize=plotsize, title='Daily Sales Standard Deviation, 7-day Rolling Average')

# Monthly Sales Percent Change
sales_monthly.pct_change().plot(figsize=plotsize, title='Monthly Sales % Change')

# Cumulative Weekly Sales
sales_weekly.cumsum().plot(figsize=plotsize, title='Cumulative Weekly Sales')

# Quarterly Sales Growth
sales_quarterly.pct_change().plot(figsize=plotsize, title='Quarterly Sales % Change')

# acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot

print('Daily data Autocorrelation Plots')
# Autocorrelation and Partial Autocorrelation Functions for Daily Data
#plot_acf(sales_new['Sales']['Furniture'])
acf_plot = plot_acf(sales_new['Furniture'], lags=30, title='Autocorrelation in Furniture Daily Sales Data')
#plot_acf(sales_new['Sales']['Furniture'])
pacf_plot = plot_pacf(sales_new['Furniture'], lags=30, title='Partial Autocorrelation in Furniture Daily Sales Data')
#plot_acf(sales_new['Sales']['Furniture'])


print('\nMonthly Data Seasonal Plot')
m_plot = month_plot(sales_monthly['Furniture'])

print('\nQuarterly Data Seasonal Plot')
q_plot = quarter_plot(sales_quarterly['Furniture'])


## Exercise 1:
# Using the source data, set up Monthly data for Sales and Profit by Segment by either (1) Resampling or (2) Grouping data by Year and Month.


### BEGIN SOLUTION
new_vars = ['Segment','Profit','Order Date','Sales']
new_base = df[new_vars].set_index('Order Date')
prof_pivot = new_base.pivot_table(columns='Segment',index = 'Order Date')
prof_month = prof_pivot.resample('M').sum()
prof_month.head()
### END SOLUTION


## Exercise 2:
# Analyze the results from the first exercise to determine whether Autocorrelation or Seasonal patterns differ by Segment or whether we are looking at Sales or Profits.

### BEGIN SOLUTION
fig,axes = plt.subplots(9,2,figsize=(20,15),)
for i,cat in enumerate(['Consumer','Corporate','Home Office']):
    for j,money in enumerate(['Sales','Profit']):
        axes[i,j].plot(prof_month[money,cat])
        axes[i,j].title.set_text(cat+" "+money)
        plot_acf(prof_month[money,cat],ax=axes[i+3,j],title = cat+" "+money+" ACF")
        month_plot(prof_month[money,cat],ax=axes[i+6,j])

fig.tight_layout()
plt.show()
### END SOLUTION


# ## Exercise 3:
# Use the result from Exercise 2 to develop an EDA function to explore other variables (like Region or Sub-Category) that may be of interest.
### BEGIN SOLUTION
cat_var = 'Region'
date_var = 'Order Date'
money_vars = ['Profit', 'Sales']

def monthly_eda(cat_var=cat_var,
                date_var=date_var, 
                money_vars=money_vars):
    new_vars = [cat_var, date_var] + money_vars
    cats = list(df[cat_var].unique())
    num_cats = len(cats)
    new_base = df[new_vars].set_index(date_var)
    prof_pivot = new_base.pivot_table(columns=cat_var,index = date_var)
    prof_month = prof_pivot.resample('M').sum()
    prof_month.head()

    fig,axes = plt.subplots(num_cats*3, 2, figsize=(20, 5*num_cats),)
    for i,cat in enumerate(cats):
        for j,money in enumerate(money_vars):
            axes[i,j].plot(prof_month[money,cat])
            axes[i,j].title.set_text(cat+" "+money)
            fig = plot_acf(prof_month[money,cat],ax=axes[i+num_cats,j],title = cat+" "+money+" ACF")
            fig = month_plot(prof_month[money,cat],ax=axes[i+num_cats*2,j])
            axes[i+num_cats*2,j].title.set_text(cat+" Seasonality")

    fig.tight_layout()
    plt.show()
### END SOLUTION

monthly_eda(cat_var='Sub-Category')

#----
# end
#----