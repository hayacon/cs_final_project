import pandas as pd #🐼


def importData(file):
    """
        Convert csv file into pandas data frame
        file : csv file
    """
    df = pd.read_csv(file)
    return df

def cleanDf(df):
    """
        Clean DataFrame by removing a row(s) with unavailble data
        df = pandas DataFrame
    """
    for index, row in df.iterrows():
        if row['comment'] == '[removed]' or row['comment'] == '[deleted]' or row['comment'] == 'this comment is no longer availble':
            df.drop(index, axis=0, inplace=True)

    return df

def dataDistribution(df):
    """
        Count a distribution of data in different range
        df : pandas Dataframe
    """
    # data score distribut
    # 1, 0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75, -1
    score_1 = 0
    score_2 = 0
    score_3 = 0
    score_4 = 0
    score_5 = 0
    score_6 = 0
    score_7 = 0
    score_8 = 0

    for index, row in df.iterrows():
        if 0.75 < row['score'] <= 1:
            score_1 = score_1 + 1
        elif 0.5 < row['score'] <= 0.75:
            score_2 = score_2 + 1
        elif 0.25 < row['score'] <= 0.5:
            score_3 = score_3 + 1
        elif 0 < row['score'] <= 0.25:
            score_4 = score_4 + 1
        elif -0.25 < row['score'] <= 0:
            score_5 = score_5 + 1
        elif -0.5 < row['score'] <= -0.25:
            score_6 = score_6 + 1
        elif -0.75 < row['score'] <= -0.5:
            score_7 = score_7 + 1
        elif -1 < row['score'] <= -0.75:
            score_8 = score_8 + 1

    data = {'-1 ~ -0.75':[score_8],
           '-0.75 ~ -0.5':[score_7],
           '-0.5 ~ -0.25':[score_6],
           '-0.25 ~ 0':[score_5],
           '0 ~ 0.25':[score_4],
           '0.25 ~ 0.5':[score_3],
           '0.5 ~ 0.75':[score_2],
           '0.75 ~ 1':[score_1]}

    score_distribution = pd.DataFrame(data=data)
    return score_distribution

    return df
