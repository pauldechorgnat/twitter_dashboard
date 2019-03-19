from happybase import Connection
import pandas as pd


def get_data_for_short_term_word_count():
    connection = Connection(host='localhost', port=9090, autoconnect=True)
    table = connection.table(name='word_count_tweets_table')
    data = table.row(row='latest')
    data = {w.decode('utf-8').split(':')[1]: int(c) for w, c in data.items()}
    return data


def get_data_for_long_term_word_count(number_of_intervals=5):
    connection = Connection(host='localhost', port=9090, autoconnect=True)
    table = connection.table(name='word_count_tweets_table')
    data = table.scan()
    total_data = {}
    for index, (row_id, word_count) in enumerate(data):
        if index == number_of_intervals:
            break
        for word, count in word_count.items():
            total_data[word.decode('utf-8')] = total_data.get(word.decode('utf-8'), 0) + int(count)

    return total_data


def get_data_for_short_term_history(number_of_intervals=15):
    connection = Connection(host='localhost', port=9090, autoconnect=True)
    table = connection.table(name='predictions_tweets_table')
    data = table.scan()
    total_data = {'volume': [], 'positive': [], 'negative': []}
    for index, (row_id, predictions) in enumerate(data):
        if index == number_of_intervals:
            break
        positive_predictions = int(predictions.get('tweet_count:positive'.encode('utf-8'), 0))
        negative_predictions = int(predictions.get('tweet_count:negative'.encode('utf-8'), 0))
        volume = positive_predictions + negative_predictions
        total_data['volume'].append(volume)
        total_data['positive'].append(positive_predictions)
        total_data['negative'].append(negative_predictions)

    return total_data


def get_data_for_long_term_history(number_of_intervals=60):
    connection = Connection(host='localhost', port=9090, autoconnect=True)
    table = connection.table(name='predictions_tweets_table')
    data = table.scan()
    total_data = {'volume': [], 'positive': [], 'negative': []}
    for index, (row_id, predictions) in enumerate(data):
        if index == number_of_intervals:
            break
        positive_predictions = int(predictions.get('tweet_count:positive'.encode('utf-8'), 0))
        negative_predictions = int(predictions.get('tweet_count:negative'.encode('utf-8'), 0))
        volume = positive_predictions + negative_predictions
        total_data['volume'].append(volume)
        total_data['positive'].append(positive_predictions)
        total_data['negative'].append(negative_predictions)

    return total_data


def get_data_for_pie_chart(number_of_intervals=60):
    data = get_data_for_long_term_history(number_of_intervals=number_of_intervals)
    data['volume'] = sum(data['volume'])
    data['positive'] = sum(data['positive'])
    data['negative'] = sum(data['negative'])
    return data


def get_data_for_tweet_table(number_of_tweets=5):
    connection = Connection(host='localhost', port=9090, autoconnect=True)
    table = connection.table(name='text_tweets_table')
    data = table.scan()
    total_data = {'time': [], 'text': [], 'authors': []}
    for index, (row_id, tweet_data) in enumerate(data):
        if index == number_of_tweets:
            break
        total_data['time'].append(tweet_data.get('metadata:publication_date'.encode('utf-8'))[:19])
        total_data['authors'].append(tweet_data.get('metadata:author'.encode('utf-8')))
        total_data['text'].append(tweet_data.get('text_data:text'.encode('utf-8')))

    return pd.DataFrame(total_data)

if __name__ == '__main__':
    print(get_data_for_tweet_table())