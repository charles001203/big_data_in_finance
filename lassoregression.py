import pandas as pd
import numpy as np
import io
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

iyear = 2004
window = 6
num_groups = 10
valid_window = 3
gridpoints = 100

filepath = '/content/joined_signals_currency.csv'
df = pd.read_csv(filepath, parse_dates=['DATE'])
df = df.dropna()
df['RET'] = df['RET']*100
df['year'] = pd.DatetimeIndex(df['DATE']).year

column_name_list = df.columns.tolist()
column_name_list.remove('mvlag')
column_name_list.remove('PERMNO')
column_name_list.remove('DATE')
column_name_list.remove('RET')
column_name_list.remove('tic')
column_name_list.remove('Date_x')
column_name_list.remove('Open_x')
column_name_list.remove('High_x')
column_name_list.remove('Low_x')
column_name_list.remove('Close_x')
column_name_list.remove('Adj Close_x')
column_name_list.remove('Volume_x')
column_name_list.remove('Date_y')
column_name_list.remove('Open_y')
column_name_list.remove('High_y')
column_name_list.remove('Low_y')
column_name_list.remove('Close_y')
column_name_list.remove('Adj Close_y')
column_name_list.remove('Volume_y')
column_name_list.remove('Date_x.1')
column_name_list.remove('Open_x.1')
column_name_list.remove('High_x.1')
column_name_list.remove('Low_x.1')
column_name_list.remove('Close_x.1')
column_name_list.remove('Adj Close_x.1')
column_name_list.remove('Volume_x.1')
column_name_list.remove('Date_y.1')
column_name_list.remove('Open_y.1')
column_name_list.remove('High_y.1')
column_name_list.remove('Low_y.1')
column_name_list.remove('Close_y.1')
column_name_list.remove('Adj Close_y.1')
column_name_list.remove('Volume_y.1')
column_name_list.remove('Date')
column_name_list.remove('Open')
column_name_list.remove('High')
column_name_list.remove('Low')
column_name_list.remove('Close')
column_name_list.remove('Adj Close')
column_name_list.remove('Volume')

print(column_name_list)

out_of_sample = df[df['year'] >= iyear + valid_window]

print(out_of_sample)

X_train = df[df['year']<2016][column_name_list]
y_train = df[df['year']<2016]['RET']
X_valid = df[(df['year']>=2016) & (df['year']<2020)][column_name_list]
y_valid = df[(df['year']>=2016) & (df['year']<2020)]['RET']
X_test = df[df['year']==2020][column_name_list]
y_test = df[df['year']==2020]['RET']
test = df[df['year']==2020]
print(X_valid)

alphas = np.linspace(0.01,500,100)
lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha');

# Lasso with 5 fold cross-validation
model = LassoCV(cv=5, random_state=0, max_iter=10000)

# Fit model
model.fit(X_train, y_train)

model.alpha_

# Set best alpha
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, y_train)

#This loop searches over values of alpha to try to maximize a version of
#the R-squared.  I set the initial alpha to be very small and the objective
#to be very negative.

alpha_opt = 0.005
obj = -100
for i in range(1,100):
    #Each iteration, I increment the candidate alpha by 0.01
    alpha_cand = i/100
    lasso = Lasso(alpha = alpha_cand)
    lfit = lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_valid)
    
    #I'm calculating an R-squared around a mean of zero rather than the
    #sample mean.  This is because the mean return is very noisy.
    u = ((y_valid-y_pred)**2).sum()
    v = ((y_valid)**2).sum()
    r2=1-u/v
    
 

    #If the R-squared from this alpha improves on the R-squared, update
    #alpha.  Otherwise keep it the same.
    if r2 > obj:
        alpha_opt=alpha_cand
        obj = r2
        

lasso = Lasso(alpha = alpha_opt)
lfit = lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

u = ((y_test-y_pred)**2).sum()
v = ((y_test)**2).sum()
r2=1-u/v
print(r2)

test['fitted_values']=y_pred

window = 0
pred_reg_expanding = []

#Now I'm going to keep doing this for an expanding window.  I have coded
#this so that you could do an expanding or rolling window.
for i in range(iyear,2021-valid_window):
    
    #Note that the training set is going to grow by one year each iteration
    #On the first iteration, it will be data from 2001-2006, the second
    #2001-2007, and so on
    
    if window > 0:
        X_train = df[(df['year']<i) & (df['year']>(i-window))][column_name_list]
        y_train = df[(df['year']<i) & (df['year']>(i-window))]['RET'].values
    else:
        X_train = df[df['year']<i][column_name_list]
        y_train = df[df['year']<i]['RET'].values
        
    X_valid = df[(df['year']>=i) & (df['year']<i+valid_window)][column_name_list]
    y_valid = df[(df['year']>=i) & (df['year']<i+valid_window)]['RET']
    
    X_test = df[df['year']==i+valid_window][column_name_list]
    y_test = df[df['year']==i+valid_window]['RET']
    
    
    alpha_opt = 0.00005
    obj = -1
    for j in range(1,100):
        alpha_cand = j/100
        lasso = Lasso(alpha = alpha_cand)
        lfit = lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_valid)
        u = ((y_valid-y_pred)**2).sum()
        v = ((y_valid)**2).sum()
        r2 = 1-u/v
        if r2 > obj:
            alpha_opt = alpha_cand
            obj = r2
            
    lasso = Lasso(alpha=alpha_opt)
    lfit = lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    
    u = ((y_test-y_pred)**2).sum()
    v = ((y_test)**2).sum()
    r2=1-u/v
    print(r2)
    
    pred = df[df['year']==i+valid_window]
    pred['fitted_values']=lasso.predict(X_test)


    test=test.append(pred)
    pred_reg_expanding.extend(pred)

window = 3
pred_reg_roll = []

for i in range(iyear,2021-valid_window):
    if window > 0:
        X_train = df[(df['year']<i) & (df['year']>=(i-window))][column_name_list]
        y_train = df[(df['year']<i) & (df['year']>=(i-window))]['RET'].values
    else:
        X_train = df[df['year']<i][column_name_list]
        y_train = df[df['year']<i]['RET'].values    
    
    X_valid = df[(df['year']>=i) & (df['year']<i+valid_window)][column_name_list]
    y_valid = df[(df['year']>=i) & (df['year']<i+valid_window)]['RET']
    
    X_test = df[df['year']==i+valid_window][column_name_list]
    y_test = df[df['year']==i+valid_window]['RET']     
  
    alpha_opt = 0.00005
    obj = -1
    for j in range(1,100):
        alpha_cand = j/100
        lasso = Lasso(alpha = alpha_cand)
        lfit = lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_valid)
        u = ((y_valid-y_pred)**2).sum()
        v = ((y_valid)**2).sum()
        r2 = 1-u/v
        if r2 > obj:
            alpha_opt = alpha_cand
            obj = r2
            
    lasso = Lasso(alpha=alpha_opt)
    lfit = lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    
    u = ((y_test-y_pred)**2).sum()
    v = ((y_test)**2).sum()
    r2=1-u/v
    print(r2)
    
    pred = df[df['year']==i+valid_window]
    pred['fitted_values']=lasso.predict(X_test)

    pred_reg_roll.extend(pred)

out_of_sample['pred_reg_expand']=np.reshape(np.array(pred_reg_expanding),(-1,1))
out_of_sample['pred_reg_roll']=np.reshape(np.array(pred_reg_roll),(-1,1))

test['ERDecile2']=test.groupby(['DATE'])['fitted_values'].\
    transform(lambda x: pd.qcut(x,num_groups, labels=False, duplicates='drop'))
test['ERDecile']=np.abs(test['ERDecile2']-(num_groups-1))

#Now, form portfolios.

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum()/ w.sum()
    except ZeroDivisionError:
        return np.nan
    
#The rest of this should hopefully be pretty familiar by now.

benchmark = test.groupby(['DATE'],as_index=False).apply(wavg,'RET','mvlag')
benchmark = benchmark.rename(columns={None:'benchmark'})    

erret_out = test.groupby(['DATE','ERDecile'],as_index=False).apply(wavg, 
                                                            'RET','mvlag')
erret_out = erret_out.rename(columns={None:'erret_out','ERDecile':'Decile'})

vwmeanret = erret_out.groupby(['Decile'])[['erret_out']].mean().reset_index()
print(vwmeanret)

erret_out=erret_out.merge(benchmark,how='inner',on='DATE')

erret_out['erp1']=1+erret_out['erret_out']/100
erret_out['rbp1']=1+erret_out['benchmark']/100

erret_out=erret_out.set_index('DATE')

erret_out['cumer']=erret_out.groupby(['Decile'])['erp1'].transform('cumprod')
erret_out['cumb']=erret_out.groupby(['Decile'])['rbp1'].transform('cumprod')

erret_out = erret_out.sort_values(['DATE','Decile'])

import matplotlib.pyplot as plt

erret_out[erret_out['Decile']==0]['cumer'].plot(label='Regression Score')
erret_out[erret_out['Decile']==0]['cumb'].plot(label='Benchmark')
plt.title('Cumulative Strategy Return')
plt.legend()
plt.show()

winner = erret_out[erret_out['Decile']==0]
winner2006=winner[0:12]
winner2006['cumer'].plot(label='Regression Score')
winner2006['cumb'].plot(label='Benchmark')
plt.title('Cumulative Strategy Return')
plt.legend()
plt.show()

print(out_of_sample.shape)
print(np.array(pred_reg_expanding).shape)
out_of_sample['pred_reg_expand']=np.array(pred_reg_expanding)
out_of_sample['pred_reg_roll']=np.reshape(np.array(pred_reg_roll),(-1,1))

out_of_sample['ERDecile1']=out_of_sample.groupby(['DATE'])['pred_reg_expand'].\
    transform(lambda x: pd.qcut(x,num_groups, labels=False, duplicates='drop'))
out_of_sample['ERDecile1']=np.abs(out_of_sample['ERDecile1']-(num_groups-1))

out_of_sample['ERDecile2']=out_of_sample.groupby(['DATE'])['pred_reg_roll'].\
    transform(lambda x: pd.qcut(x,num_groups, labels=False, duplicates='drop'))
out_of_sample['ERDecile2']=np.abs(out_of_sample['ERDecile2']-(num_groups-1))

#rf = pd.read_csv('../../Data/Class4/rf.csv',parse_dates=['DATE'])
rf = pd.read_csv('rf.csv',parse_dates=['DATE'])
winner = port_ret[port_ret['Decile']==0]
winner = winner.merge(rf,how='inner',on='DATE')
winner['rfp1']=1+winner['rf']
winner['cumrf']=winner['rfp1'].transform('cumprod')

import statsmodels.formula.api as sm

newreg = sm.ols('erret_out~benchmark',data=winner).fit()
beta = newreg.params[1]

er_comp = winner['cumer'][len(winner)-1]**(1/(2021-iyear))-1
br_comp = winner['cumb'][len(winner)-1]**(1/(2021-iyear))-1
rf_comp = winner['cumrf'][len(winner)-1]**(1/(2021-iyear))-1

alpha = er_comp - rf_comp - beta*(br_comp-rf_comp)

print("Strategy Compounded Return: {}".format(er_comp))
print("Benchmark Compounded Return: {}".format(br_comp))
print("Alpha: {}".format(alpha))

drawdown1 = []
drawdown3 = []
drawdown12 = []
for i in range(12,len(winner)):
    retm = winner['cumer'][i]/winner['cumer'][i-1]-1
    retq = winner['cumer'][i]/winner['cumer'][i-3]-1
    reta = winner['cumer'][i]/winner['cumer'][i-12]-1
    drawdown1 = np.append(drawdown1,retm)
    drawdown3 = np.append(drawdown3,retq)
    drawdown12 = np.append(drawdown12, reta)
    
print("Maximum 1-Month Drawdown: {}".format(min(drawdown1)))
print("Maximum 3-Month Drawdown: {}".format(min(drawdown3)))
print("Maximum 12-Month Drawdown: {}".format(min(drawdown12)))