import data_analysis_functions as da
import pandas as pd #🐼
from transformers import AutoTokenizer
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

#filter email
def filter_emails(text):
    pattern = r'(?:(?!.*?[.]{2})[a-zA-Z0-9](?:[a-zA-Z0-9.+!%-]{1,64}|)|\"[a-zA-Z0-9.+!% -]{1,64}\")@[a-zA-Z0-9][a-zA-Z0-9.-]+(.[a-z]{2,}|.[0-9]{1,})'
    text = re.sub(pattern, '', text)
    return text

#filter website url
def filter_websites(text):
    pattern = r'(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z][a-z\-]*'
    text = re.sub(pattern, '', text)
    return text

def clean_text(text):
    #initialize WordNet lemmatizer from NLTK
    lemmatizer = WordNetLemmatizer()
    #tokenize text
    tokens = word_tokenize(text)
    # convert to lower case
    text = [w.lower() for w in tokens]
    #remove punctuation
    table = str.maketrans('', '', string.punctuation)
    text = [w.translate(table) for w in text]
    # remove remaining tokens that are not alphabetic
    text = [word for word in text if word.isalpha()]
    #lemmatization
    text = [lemmatizer.lemmatize(i) for i in text]
    #put tokens back together
    text = untokenize(text)
    #remove emails
    text = filter_emails(text)
    #remove urls
    text = filter_websites(text)

    return text

def textConvert(df, col, new_col):
    '''
    Clean original text and store it in a new column
    df : pandas DataFrame
    col : column contain original text
    new_col : new column to contain cleaned text
    '''
    df.insert(2, new_col, 'abc')
    for index, row in df.iterrows():
        df.iat[index, 2] = clean_text(row[col])
    return df

def data_vectorization(df):
    # df = da.importData(file)
    # clean_df = da.cleanDf(df)
    # reddit_df = textConvert(clean_df, 'comment', 'cleaned_comment')

    # for index, row in reddit_df.iterrows():
    #     reddit_df.iat[index, 2] = untokenize(row['cleaned_comment'])

    #install hateBERT tokenizor
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

    #fix max_length since there are way too many unnecessary zeros...
    encoded_corpus = tokenizer(text=df.cleaned_comment.tolist(),
                                add_special_tokens=True,
                                padding='max_length',
                                truncation='longest_first',
                                max_length=300,
                                return_attention_mask=True)
    input_ids = encoded_corpus['input_ids']
    attention_mask = encoded_corpus['attention_mask']
    df['encoded_comment'] = input_ids
    df['encoding_mask'] = attention_mask

    return df

def test_train_dataset(df):
    # shuffle a dataset
    df = df.sample(frac=1).reset_index(drop=True)

    train_df = df.iloc[:4604,:]
    test_df = df.iloc[4605:,:]

    train_df = train_df.dropna()
    trst_df = test_df.dropna()

    train_df.to_csv('data/training_set.csv')
    test_df.to_csv('data/test_set.csv')

    return test_df, train_df
