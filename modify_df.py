import pandas as pd 

# For October 2017 data
def get_oct17_data(df, column_name):
    sample = df[column_name].tolist()
    years = []
    for s in sample:
        if isinstance(s, float): # Skip instance if it is a float
            continue
        years.append(s[-4:]) # Keep only the last 4 characters of s
    df = df.drop(columns = ['index', column_name])
    df_years = pd.DataFrame(years, columns = ['Years'])
    df = pd.concat([df, df_years], axis = 1)
    df = df.rename(columns = {'text':'Tweet'})
    return df

# For November 2017 - December 2017 data
def get_novdec17_data(df, column_name):
    sample = df[column_name].tolist()
    years = []
    for s in sample:
        years.append('2017')
    df = df.drop(columns = ['index', column_name])
    df_years = pd.DataFrame(years, columns = ['Years'])
    df = pd.concat([df, df_years], axis = 1)
    df = df.rename(columns = {'text':'Tweet'})
    return df

# For September 2018 - February 2019 data
def get_sept18feb19_data(df, column_name):
    sample = df[column_name].tolist()
    years = []
    for s in sample:
        if "2018" in s:
            years.append('2018')
        elif "2019" in s:
            years.append('2019')
    df = df.drop(columns = ['index', column_name])
    df_years = pd.DataFrame(years, columns = ['Years'])
    df = pd.concat([df, df_years], axis = 1)
    df = df.rename(columns = {'text':'Tweet'})
    return df

# For October 2019 data
def get_oct19_data(df, column_name):
    sample = df[column_name].tolist()
    years = []
    for s in sample:
        years.append('2019')
    df = df.drop(columns = ['index', column_name])
    df_years = pd.DataFrame(years, columns = ['Years'])
    df = pd.concat([df, df_years], axis = 1)
    df = df.rename(columns = {'Text':'Tweet'})
    return df