import json
import re
import random
import datetime
from collections import Counter
from tqdm import tqdm
from happybase import Connection

thrift_server_host = 'localhost'
thrift_server_port = 9090
path_to_tweet_file = './large_tweets3.json'
path_to_stop_words_file = './stopwords.txt'

if __name__ == '__main__':
    # creating a connection to the table
    connection = Connection(host=thrift_server_host, port=thrift_server_port, autoconnect=True)
    # name of our table
    table_name = 'short_term_tweet_dashboard'
    # meta data of our table
    column_families = {
        'metadata':dict(),  # this column family will be used to store the datetime
        'tweet_text': dict(),  # this column family will contain all the text
        'predictions': dict(),  # this column family will contain the predictions (probability and label)
        'word_count': dict(),  # this column family will contain the word counts
    }

    # creating a table if it does not exist
    if table_name.encode('utf-8') in connection.tables():
        connection.disable_table(name=table_name)
        connection.delete_table(name=table_name)
    connection.create_table(name=table_name, families=column_families)
    # getting the table
    table = connection.table(name=table_name)

    # loading tweets
    with open(path_to_tweet_file, 'r') as file:
        tweets = json.load(file)

    # loading stop words
    stop_words = []
    with open(path_to_stop_words_file) as file:
        for line in file:
            stop_words.append(line.strip())
    # turning the list of stop words into a set
    stop_words = set(stop_words + ['trump'])

    # creating a regular expression for words
    word_regex = re.compile('\\w+')

    # preparing to go over the data
    total_number_of_tweets = len(tweets)
    number_of_windows = 100
    number_of_tweets_per_batch = total_number_of_tweets // number_of_windows
    time = datetime.datetime.now()
    delta = datetime.timedelta(seconds=10)

    for i in tqdm(range(number_of_windows)):
        tweets_batch = tweets[i * number_of_tweets_per_batch:(i + 1) * number_of_tweets_per_batch]
        total_text = ''
        data_to_insert = {'metadata:datetime': str(time)[:19]}
        for index_tweet, tweet in enumerate(tweets_batch):
            text = tweet['text']
            total_text += ' ' + text.lower()
            data_to_insert['tweet_text:tweet_{}'.format(index_tweet)] = text
            score = random.uniform(-1, 1)
            data_to_insert['predictions:tweet_{}_score'.format(index_tweet)] = str(score)
            data_to_insert['predictions:tweet_{}_label'.format(index_tweet)] = str((score > 0) * 1)
        words = [w for w in word_regex.findall(total_text) if w not in stop_words]
        word_count = Counter(words)
        for word in sorted(word_count, key=word_count.get, reverse=True)[:100]:
            data_to_insert['word_count:{}'.format(word)] = str(word_count[word])
        table.delete(row='latest')
        table.put(row='latest', data=data_to_insert)
        table.put(row=str(time)[:19], data=data_to_insert)
        time += delta

    connection.close()
