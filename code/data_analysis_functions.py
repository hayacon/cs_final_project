import pandas as pd #🐼
import errno
import matplotlib.pyplot as plt
import re
import string

#Natul Language Toollit
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords

from wordcloud import WordCloud

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('omw-1.4')

def importData(file):
    """
    Convert csv file into pandas data frame
    file : csv file
    """
    try:
        df = pd.read_csv(file, index_col=None)
    except IOError as e:
        if e.errno == errno.EACCES:
            print("file exists, but isn't readable")
        elif e.errno == errno.ENOENT:
            print("files isn't readable because it isn't there")

    df.drop(columns='Unnamed: 0', inplace=True)
    return df

def cleanDf(df):
    """
    Clean DataFrame by removing a row(s) with unavailble data
    df = pandas DataFrame
    """
    for index, row in df.iterrows():
        if row['comment'] == '[removed]' or row['comment'] == '[deleted]' or row['comment'] == 'this comment is no longer availble':
            df.drop(index, axis=0, inplace=True)
    df = df.reset_index()
    df = df.drop(columns='index')
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

def cleanText(text_data):
    """
    Helper function to clean text data by remove all unnecessary words, and with lemmatization.
    The following items are remove : URLs, stop words, a single character words,
    punctuations, non alphabetic words.
    text_data : column of dataframe
    """
    #initialize WordNet lemmatizer from NLTK
    lemmatizer = WordNetLemmatizer()
    # for text in text_data:
    #remove all urls
    text = re.sub(r"http\S+", "", text_data)
    text = word_tokenize(text)
    # convert to lower case
    text = [w.lower() for w in text]
    #remove punctuations
    table = str.maketrans('', '', string.punctuation)
    text = [w.translate(table) for w in text]
    # remove all tokens that are not alphabetic
    text = [word for word in text if word.isalpha()]
    # filter out short tokens
    text = [word for word in text if len(word) > 1]
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [i for i in text if not i in stop_words]
    #lemmatization
    text = [lemmatizer.lemmatize(i) for i in text]

    return text

def flatten(df, col):
    """
    helper function to convert a cleaned_text column into a single list
    df : dataframe
    """
    t = df[col].tolist()
    return [item for sublist in t for item in sublist]

def textConvert(df, col, new_col):
    '''
    Clean original text and store it in a new column
    df : pandas DataFrame
    col : column contain original text
    new_col : new column to contain cleaned text
    '''
    df.insert(2, new_col, 'abc')
    for index, row in df.iterrows():
        df.iat[index, 2] = cleanText(row[col])
    return df

def lexical_diversity(text):
    '''
    Function to calculate a number of vocab and lexical diversity
    text : a list of words
    '''
    vocab = len(set(text))
    lex = vocab/len(text)
    return vocab, lex

def plot_lexical_diversity(vocab, lex, index):
    '''
    Function to plot a table of vacaburary and lexical diversity of text data
    vocab : array of numerical data (size of vocaburay)
    lex : array of lexical diversity of several data
    '''
    if len(vocab) != len(lex):
        print("!size of vaoburay and lexical diversity must be equal!")
        return None

    d = {'Vocaburaly':vocab, 'Lexical Diversity':lex}
    df = pd.DataFrame(data=d, index=index)
    return df

def frequentDistribution(text):
    return FreqDist(text)

def plot_wordCloud(text):
    """
    helper function to plot word cloud
    text : a list of words
    """
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(str(text))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    return plt
