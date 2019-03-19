import json
from kafka import SimpleClient, SimpleProducer
import time
import random
from argparse import ArgumentParser
import requests
import os


def download_file_from_google_drive(id_, destination_):
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': id_}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id_, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination_)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination_):
    chunk_size = 32768

    with open(destination_, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':

    # getting the path to the json file
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--path',
                            type=str,
                            help='path to the json file containing the tweets',
                            default='/home/paul/data_engineering/data/large_tweets.json')

    # parsing arguments
    arguments = arg_parser.parse_args()

    # getting the path to the json file containing tweets
    path_to_tweets = arguments.path
    # getting the parent folder
    path_to_parent_folder = '/'.join(path_to_tweets.split('/')[:-1])

    # checking that the file exists
    if path_to_tweets.split('/')[-1] not in os.listdir(path_to_parent_folder):

        print('starting dowloading data')

        # downloading the data from Google Drive
        file_id = '1sOXurkOh48nT7AuuTqN80jf05D13WFld'
        destination = path_to_tweets
        download_file_from_google_drive(file_id, destination)

        print('downloading ended')

    # creating a kafka client
    kafka_client = SimpleClient(hosts='localhost:9092')

    # creating a kafka producer
    kafka_producer = SimpleProducer(client=kafka_client)

    # opening the file containing the tweets
    with open(path_to_tweets, 'r') as file:
        tweets = json.load(file)

    for index_tweet, tweet in enumerate(tweets):
        time.sleep(random.uniform(0, 1)/10)
        kafka_producer.send_messages('trump', str(tweet).encode('utf-8'))
        if index_tweet+1 % 100 == 0:
            print('emitting tweet number {} over {}'.format(index_tweet, len(tweets)))
