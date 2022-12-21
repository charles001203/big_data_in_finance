import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import io
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

iyear = 2010
window = 6
num_groups = 10
valid_window = 3
gridpoints = 100

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum()/ w.sum()
    except ZeroDivisionError:
        return np.nan

filepath = '/content/signals1.csv'
df = pd.read_csv('smoothed_data_wNA.csv', parse_dates=['DATE'])
df = df.dropna()
df['smooth_ret'] = df['smooth_ret']*100
df.drop(columns=['RET'])
df['year'] = pd.DatetimeIndex(df['DATE']).year

column_name_list = df.columns.tolist()

column_name_list.remove('mvlag')
column_name_list.remove('PERMNO')
column_name_list.remove('DATE')
column_name_list.remove('RET')
column_name_list.remove('smooth_ret')
column_name_list.remove('tic')
column_name_list.remove('Date_x')
column_name_list.remove('Open_x')
column_name_list.remove('High_x')
column_name_list.remove('Low_x')
column_name_list.remove('Close_x')
column_name_list.remove('Adj.Close_x')
column_name_list.remove('Volume_x')
column_name_list.remove('Date_y')
column_name_list.remove('Open_y')
column_name_list.remove('High_y')
column_name_list.remove('Low_y')
column_name_list.remove('Close_y')
column_name_list.remove('Adj.Close_y')
column_name_list.remove('Volume_y')
column_name_list.remove('Date_x.1')
column_name_list.remove('Open_x.1')
column_name_list.remove('High_x.1')
column_name_list.remove('Low_x.1')
column_name_list.remove('Close_x.1')
column_name_list.remove('Adj.Close_x.1')
column_name_list.remove('Volume_x.1')
column_name_list.remove('Date_y.1')
column_name_list.remove('Open_y.1')
column_name_list.remove('High_y.1')
column_name_list.remove('Low_y.1')
column_name_list.remove('Close_y.1')
column_name_list.remove('Adj.Close_y.1')
column_name_list.remove('Volume_y.1')
column_name_list.remove('Date')
column_name_list.remove('Open')
column_name_list.remove('High')
column_name_list.remove('Low')
column_name_list.remove('Close')
column_name_list.remove('Adj.Close')
column_name_list.remove('Volume')

print(column_name_list)

out_of_sample = df[df['year'] >= iyear + valid_window]

window = 0
pred_reg_expanding = []

#Expanding window regression
for i in range(iyear,2021-valid_window):    
    if window > 0:
        X_train = df[(df['year']<i) & (df['year']>=(i-window))][column_name_list]
        y_train = df[(df['year']<i) & (df['year']>=(i-window))]['smooth_ret'].values

    else:
        X_train = df[df['year']<i][column_name_list]
        y_train = df[df['year']<i]['smooth_ret'].values    
    
    X_valid = df[(df['year']>=i) & (df['year']<i+valid_window)][column_name_list]
    y_valid = df[(df['year']>=i) & (df['year']<i+valid_window)]['smooth_ret']
    
    X_test = df[df['year']==i+valid_window][column_name_list]
    y_test = df[df['year']==i+valid_window]['smooth_ret']   

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    y_pred_reg = reg.predict(X_test)  

    pred_reg_expanding.extend(y_pred_reg)

window = 3
pred_reg_roll = []

for i in range(iyear,2021-valid_window):
    if window > 0:
        X_train = df[(df['year']<i) & (df['year']>=(i-window))][column_name_list]
        y_train = df[(df['year']<i) & (df['year']>=(i-window))]['smooth_ret'].values
    else:
        X_train = df[df['year']<i][column_name_list]
        y_train = df[df['year']<i]['smooth_ret'].values    
    
    X_valid = df[(df['year']>=i) & (df['year']<i+valid_window)][column_name_list]
    y_valid = df[(df['year']>=i) & (df['year']<i+valid_window)]['smooth_ret']
    
    X_test = df[df['year']==i+valid_window][column_name_list]
    y_test = df[df['year']==i+valid_window]['smooth_ret']     
  
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    print(reg.score(X_valid, y_valid))
    y_pred_reg = reg.predict(X_test)

    pred_reg_roll.extend(y_pred_reg)

out_of_sample['pred_reg_expand']=np.reshape(np.array(pred_reg_expanding),(-1,1))
out_of_sample['pred_reg_roll']=np.reshape(np.array(pred_reg_roll),(-1,1))

out_of_sample['ERDecile1']=out_of_sample.groupby(['DATE'])['pred_reg_expand'].\
    transform(lambda x: pd.qcut(x,num_groups, labels=False, duplicates='drop'))
out_of_sample['ERDecile1']=np.abs(out_of_sample['ERDecile1']-(num_groups-1))

out_of_sample['ERDecile2']=out_of_sample.groupby(['DATE'])['pred_reg_roll'].\
    transform(lambda x: pd.qcut(x,num_groups, labels=False, duplicates='drop'))
out_of_sample['ERDecile2']=np.abs(out_of_sample['ERDecile2']-(num_groups-1))

benchmark = out_of_sample.groupby(['DATE'],as_index=False).apply(wavg,'RET','mvlag')
benchmark = benchmark.rename(columns={None:'benchmark'})   

erret1 = out_of_sample.groupby(['DATE','ERDecile1'],as_index=False).apply(wavg, 
                                                            'RET','mvlag')
erret1 = erret1.rename(columns={None:'erret1','ERDecile1':'Decile'})

erret2 = out_of_sample.groupby(['DATE','ERDecile2'],as_index=False).apply(wavg, 
                                                            'RET','mvlag')
erret2 = erret2.rename(columns={None:'erret2','ERDecile2':'Decile'})

port_ret = benchmark.merge(erret1, how='inner', on='DATE')
port_ret = port_ret.merge(erret2, how='inner', on=['DATE','Decile'])

port_ret['rb_gross']=1+port_ret['benchmark']/100
port_ret['r1_gross']=1+port_ret['erret1']/100
port_ret['r2_gross']=1+port_ret['erret2']/100

port_ret=port_ret.set_index('DATE')

port_ret['rcumb'] = port_ret.groupby(['Decile'])['rb_gross'].transform('cumprod')
port_ret['rcum1'] = port_ret.groupby(['Decile'])['r1_gross'].transform('cumprod')
port_ret['rcum2'] = port_ret.groupby(['Decile'])['r2_gross'].transform('cumprod')

port_ret = port_ret.sort_values(['DATE','Decile'])

port_ret[port_ret['Decile']==0]['rcumb'].plot(label='Benchmark')
port_ret[port_ret['Decile']==0]['rcum1'].plot(label='Regression Expanding')
port_ret[port_ret['Decile']==0]['rcum2'].plot(label='Regression Rolling')
plt.title('Cumulative Strategy Return')
plt.legend()
plt.show()

winner = port_ret[port_ret['Decile']==0]
rf = pd.read_csv('rf.csv',parse_dates=['DATE'])
winner = winner.merge(rf,how='inner',on='DATE')
winner['rfp1']=1+winner['rf']
winner['cumrf']=winner['rfp1'].transform('cumprod')

newreg1 = sm.ols('erret1~benchmark',data=winner).fit()
beta1 = newreg1.params[1]

newreg2 = sm.ols('erret2~benchmark',data=winner).fit()
beta2 = newreg2.params[1]

geom_avg1 = winner['rcum1'][len(winner)-1]**(1/(2021-iyear))-1
geom_avg2 = winner['rcum2'][len(winner)-1]**(1/(2021-iyear))-1
geom_avgb = winner['rcumb'][len(winner)-1]**(1/(2021-iyear))-1
geom_avgf = winner['cumrf'][len(winner)-1]**(1/(2021-iyear))-1

alpha1 = geom_avg1 - geom_avgf - beta1*(geom_avgb-geom_avgf)
alpha2 = geom_avg2 - geom_avgf - beta2*(geom_avgb-geom_avgf)

print("Strategy Compounded Return: {}".format([geom_avg1,geom_avg2]))
print("Benchmark Compounded Return: {}".format(geom_avgb))
print("Alpha: {}".format([alpha1,alpha2]))
print("R-squared: {}".format([newreg1.rsquared_adj, newreg2.rsquared_adj]))

drawdown1 = []
drawdown3 = []
drawdown12 = []
for i in range(12,len(winner)):
    retm = winner['cumrf'][i]/winner['cumrf'][i-1]-1
    retq = winner['cumrf'][i]/winner['cumrf'][i-3]-1
    reta = winner['cumrf'][i]/winner['cumrf'][i-12]-1
    drawdown1 = np.append(drawdown1,retm)
    drawdown3 = np.append(drawdown3,retq)
    drawdown12 = np.append(drawdown12, reta)
    
print("Maximum 1-Month Drawdown: {}".format(min(drawdown1)))
print("Maximum 3-Month Drawdown: {}".format(min(drawdown3)))
print("Maximum 12-Month Drawdown: {}".format(min(drawdown12)))

print(winner)

column_name_list.remove('column_label')
column_name_list.remove('X')
column_name_list.remove('group')
column_name_list.remove('year')

ols_formula = 'smooth_ret~' + "+".join(column_name_list)

cluster_date_ols = sm.ols(formula = ols_formula,
                        data=df).fit(cov_type='cluster',
                        cov_kwds={'groups': df['DATE']},
                        use_t=True)

print(cluster_date_ols.summary())