import pandas as pd
def main():
    currency_AED = pd.read_csv('CurrencyData/AEDUSD=X.csv')
    currency_CAD = pd.read_csv('CurrencyData/CADUSD=X.csv')
    currency_RUB = pd.read_csv('CurrencyData/RUBUSD=X.csv')
    currency_SAR = pd.read_csv('CurrencyData/SARUSD=X.csv')
    currency_IQD = pd.read_csv('CurrencyData/IQDUSD=X.csv')


    #calculate the daily return
    volitality_AED = currency_AED["Open"].pct_change(fill_method='ffill')
    volitality_CAD = currency_CAD["Open"].pct_change(fill_method='ffill')
    volitality_RUB = currency_AED["Open"].pct_change(fill_method='ffill')
    volitality_SAR = currency_AED["Open"].pct_change(fill_method='ffill')
    volitality_IQD = currency_IQD["Open"].pct_change(fill_method='ffill')

    currency_AED["volitality_AED"] = volitality_AED
    #get rid of dash
    modified_dates = []
    for i in range(0,len(currency_AED)):
       modified_dates.append(int(currency_AED.loc[i].at["Date"].replace('-','')))
    currency_AED["Date"] = modified_dates

    currency_CAD["volitality_CAD"] = volitality_CAD
    #get rid of dash
    modified_dates = []
    for i in range(0,len(currency_CAD)):
       modified_dates.append(int( currency_CAD.loc[i].at["Date"].replace('-','')))
    currency_CAD["Date"] = modified_dates

    currency_RUB["volitality_RUB"] = volitality_RUB
    #get rid of dash
    modified_dates = []
    for i in range(0,len(currency_RUB)):
       modified_dates.append(int(currency_RUB.loc[i].at["Date"].replace('-','')))
    currency_RUB["Date"] = modified_dates

    currency_SAR["volitality_SAR"] = volitality_SAR
    #get rid of dash
    modified_dates = []
    for i in range(0,len(currency_SAR)):
       modified_dates.append(int(currency_SAR.loc[i].at["Date"].replace('-','')))
    currency_SAR["Date"] = modified_dates

    currency_IQD["volitality_IQD"] = volitality_IQD
    #get rid of dash
    modified_dates = []
    for i in range(0,len(currency_IQD)):
       modified_dates.append(int(currency_IQD.loc[i].at["Date"].replace('-','')))
    currency_IQD["Date"] = modified_dates


    #inner join on our dataframe by time
    signals = pd.read_csv("CurrencyData/signals1.csv")
    modified_dates = []
    for i in range(0,len(signals)):
        modified_dates.append(int(signals.loc[i].at["DATE"]))
    signals["DATE"] = modified_dates

    joined_data = pd.merge(signals, currency_AED, left_on=['DATE'], right_on= ['Date'], how='inner')
    joined_data = pd.merge(joined_data, currency_CAD, left_on=['DATE'], right_on=['Date'], how='inner')
    joined_data = pd.merge(joined_data, currency_RUB, left_on=['DATE'], right_on=['Date'], how='inner')
    joined_data = pd.merge(joined_data, currency_SAR, left_on=['DATE'], right_on=['Date'], how='inner')
    joined_data = pd.merge(joined_data, currency_IQD, left_on=['DATE'], right_on=['Date'], how='inner')
    joined_data.to_csv("joined_signals_currency.csv")



if __name__ == "__main__":
    main()