from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, Row
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from nltk.corpus import stopwords
import re
import json


# directory_path = 'file:///home/hduser/twitter_data'
# file_path = '/data/twitter_data/z_sample.csv'
file_path = 'file:///home/paul/data_engineering/data/sample.csv'


def tokenize(text, common_words):
    """
    function used to tokenize tweets
    """
    word_re = '\\w+'
    pseudo_re = '\\@[a-zA-Z0-9_]*'
    link_re = 'http(s)?://[a-zA-Z0-9./\\-]*'

    # replacing pseudos
    text1 = re.sub(pattern=pseudo_re, string=text, repl='pseudotwitterreplacement')
    # replacing links
    text1 = re.sub(pattern=link_re, string=text1, repl='linkwebsitereplacement')
    # replacing RTs
    text1 = re.sub(pattern='RT', string=text1, repl='retweetreplacement')

    # finding tokens
    tokens = [t for t in re.findall(pattern=word_re, string=text1.lower()) if t in common_words]

    return tokens


def load_stopwords():
    """
    function used to load stop words
    """
    # trying to load the english stop words
    try:
        stopwords.words('english')
    except LookupError:
        # if an error occurs, we download the stopwords with nltk
        import nltk
        nltk.download('stopwords')
    finally:
        # creating a set
        sw = set(stopwords.words('english'))
        # we want to keep some of the words because they can have a lot of sense
        words_to_keep = {'no', 'not', 'up', 'off', 'down', 'yes'}
        # removing those words from the list of stop words
        sw = sw.difference(words_to_keep)
    return sw


def load_common_words(directory='/home/paul/data_engineering/data'):
    """
    function used to load most common words
    """
    # loading the most common words
    cm = set(open('{}/most_used_words.csv'.format(directory)).read().split('\n'))
    # adding our placeholders
    cm.update(['pseudotwitterreplacement', 'linkwebsitereplacement', 'retweetreplacement'])
    return cm


def create_hash_table(common_words, stop_words):
    """
    function used to create a hash table of the words we want to keep
    """
    # deleting the stop words
    words = common_words.difference(stop_words)
    return {w: i for i, w in enumerate(words)}


def compute_tf(tokens, reference_table):
    """function used to compute term frequency"""
    hash_table = {}
    # running through the tokens
    for token in tokens:
        # if the token is indeed among those we want to keep
        if token in reference_table.keys():
            # updating the frequency table
            hash_table[reference_table[token]] = hash_table.get(reference_table[token], 0) + 1
    # returning a Sparse vector object
    sparse_vector = SparseVector(len(reference_table), hash_table)
    return sparse_vector


if __name__ == '__main__':

    # create a spark context
    spark_context = SparkContext.getOrCreate()
    sql_context = SQLContext(sparkContext=spark_context)

    # defining the schema of the data
    schema = StructType([
        StructField('sentiment', IntegerType(), True),
        StructField('text', StringType(), True)
    ])

    # load data
    df = sql_context.read.csv(path=file_path, schema=schema, header=True)
    
    # using a rdd
    rdd = df.rdd.map(Row.asDict)

    # getting the stop words
    sw = load_stopwords()

    cm = load_common_words()

    reference_table = create_hash_table(common_words=cm, stop_words=sw)


    # tokenizing the text
    rdd = rdd.map(lambda d:
                  {
                      'tokens': tokenize(text=d['text'], common_words=cm),
                      'label': d['sentiment']
                  }).\
        map(lambda d: LabeledPoint(0 if d['label'] == 0 else 1,
                                   compute_tf(tokens=d['tokens'],
                                              reference_table=reference_table)))
    # instantiating the logistic regression
    logistic_regression = LogisticRegressionWithSGD()
    # training the logistic regression
    trained_logistic_regression = logistic_regression.train(data=rdd)

    # storing the parameters in a json file
    trained_parameters = {
        'weights': trained_logistic_regression.weights.toArray().tolist(),
        'intercept': trained_logistic_regression.intercept
    }

    with open('/home/paul/data_engineering/data/model.json', 'w') as model_file:
        json.dump(trained_parameters, fp=model_file)
