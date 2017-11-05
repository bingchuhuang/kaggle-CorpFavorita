import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import math

df_comb = pd.read_csv('combine_data_2017-06-15_to_2017-08-15.csv')

df_filter = df_comb[(df_comb['family'] == 8)]
sales_f0 = df_filter['unit_sales']
date_f0 = pd.to_datetime(df_filter['date'])
oil_f0 = df_filter['dcoilwtico']

grouped = df_filter.groupby('date')
sales_mean = []
sales_std = []
sales_date = []
for date, group in grouped:
    mean = group['unit_sales'].mean()
    std  = group['unit_sales'].std()
    sales_date.append(pd.to_datetime(date))
    sales_mean.append(mean)
    sales_std.append(std)

fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
sns.regplot(x=oil_f0, y=sales_f0, x_bins=10)
fig.tight_layout()

plt.savefig('feature_fig/sales_vs_oil.png')

fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
fig.autofmt_xdate()
ax.errorbar(sales_date, sales_mean, yerr = sales_std, marker='s')
fig.tight_layout()
plt.savefig('feature_fig/sales_vs_date.png')


df_filter = df_filter.fillna(-1)
grouped = df_filter.groupby('type') #holiday type
sales_holiday_mean = []
sales_holiday_std = []
sales_holiday_type = []
for type, group in grouped:
    print(type)
    mean = group['unit_sales'].mean()
    std  = group['unit_sales'].std()
    if math.isnan(type) :
        sales_holiday_type.append(-1)
    else:
        sales_holiday_type.append(type)

    sales_holiday_mean.append(mean)
    sales_holiday_std.append(std)

fig, ax = plt.subplots()
fig.set_size_inches(6, 6)
ax.errorbar(sales_holiday_type, sales_holiday_mean, yerr = 0, marker='s')
#ax.errorbar(sales_holiday_type, sales_holiday_mean, yerr = sales_holiday_std, marker='s')
fig.tight_layout()
plt.savefig('feature_fig/sales_vs_holiday.png')

plt.show()
