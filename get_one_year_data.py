import pandas as pd

df = pd.read_csv('rank_signals8.csv')
end_df = pd.DataFrame()

uniq_permno = df.permno.unique()

for i in range(2001,2021):
    temp_df = df[df['year'] == i]
    df2 = pd.DataFrame()
    
    #go through each permno
    for j in uniq_permno:
        try:
            temp_df1 = temp_df[temp_df['permno'] == j]
            df2 = temp_df1.iloc[0]
            ret = 1

            #get the cumulative return over the year
            for k in range(0, len(temp_df1)):
                ret *= (1+temp_df['RET'].iloc[k])

            ret -= 1
            df2['RET'] = ret
        
            end_df = end_df.append(df2)
        except:
            pass

print(end_df)

#Output to a seperate file
# end_df.to_csv('1_year_data.csv')