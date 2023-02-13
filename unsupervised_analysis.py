
import argparse
import csv
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import torch

from intent_classification_tfidf import remove_diacritics
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from bertopic import BERTopic


# Preprocessing function to clean the tweets
def preprocess_tweet(tweet):
    # remove handles, hashtags, urls
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    # remove urls
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'www\S+', '', tweet)
    tweet = re.sub(r'pic.twitter\S+', '', tweet)

    tweet = re.sub(r'\W', ' ', tweet) # remove special characters
    tweet = re.sub(r'\s+', ' ', tweet) # remove multiple whitespaces

    # tweet = remove_diacritics(tweet)

    tweet = tweet.strip()

    return tweet


# Plot t-SNE visualization of the embedding matrix
def plot_tsne(embedding_matrix, labels, model_tag):
    tsne = TSNE(n_components=2, perplexity=30, verbose=2)
    tsne_results = tsne.fit_transform(embedding_matrix)

    # Plot each cluster with a different color efficiently
    for i, label in enumerate(set(labels)):
        plt.scatter(tsne_results[labels == label, 0], tsne_results[labels == label, 1], label=label)
    plt.legend()
    plt.savefig('tsne-{}.png'.format(model_tag))


# Load tweets from json
def load_tweets_from_json(json_file, cased=False, preprocess=True):
    with open(json_file, 'r') as f:
        json_ = json.load(f)

    tweets = []
    for i, tweet_ in enumerate(json_):
        tweet = tweet_['full_text']
        if preprocess:
            tweet = preprocess_tweet(tweet)
        if not cased:
            tweet = tweet.lower()  # convert to lowercase
        tweets.append(tweet)

    return tweets


def main():
    # Add arguments for model tag and json filepath
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_tag', type=str, default='')
    parser.add_argument('--json_file', type=str, default='postgres_public_feeds_entry.json')
    parser.add_argument('--preprocess', action="store_true",
                        help="Assume tweets are preprocessed if this flag is not set.")
    args = parser.parse_args()

    model_tag = args.model_tag
    cased = True if 'cased' in model_tag else False
    model = SentenceTransformer(model_tag) if model_tag else None
    tweets = load_tweets_from_json(args.json_file, cased=cased, preprocess=args.preprocess)
    tweets = list(set(tweets))
    if args.preprocess:
        # Save the preprocessed tweets to a json file in format "{"full_text": tweet}
        # to be used in the intent classification script"
        if cased:
            out_json = 'postgres_public_feeds_entry_preprocessed_tr.json'
        else:
            out_json = 'postgres_public_feeds_entry_preprocessed_tr_lower.json'
        with open(out_json, 'w') as f:
            json.dump([{"full_text": tweet} for tweet in tweets], f)

    keywords = {
        'kurtarma': 'enkaz altında yardım edin kurtarın bekliyor',
        'ses': 'ses geliyor seslerini duyuyoruz ağlama sesleri geliyor',
        'yemek_su': 'aç susuz yemek lazım gıda ihtiyacı var içme suyu eksik',
        'giysi': 'battaniye çadır ve sıcak tutan giysiler gerekiyor çok soğuk',
        'saglik': 'yaralılar var hastalara doktor ve ilaç lazım tedavi gerekli',
        'siyaset': 'siyaset yapmanın sırası değil siyasetiniz batsın'
    }
    # Convert keywords to a list of lists by splitting on whitespace
    for k, v in keywords.items():
        keywords[k] = v.split(' ')

    topic_model = BERTopic(
        embedding_model=model,
        language='multilingual',
        seed_topic_list=list(keywords.values()),
        n_gram_range=(1, 2),
        min_topic_size=50,
        nr_topics=50,
        verbose=True)
    topics, _ = topic_model.fit_transform(tweets)

    for k, v in topic_model.topic_sizes_.items():
        print("{}: {}".format(v, topic_model.topic_labels_[k]))

    print()


if __name__ == "__main__":
    main()
