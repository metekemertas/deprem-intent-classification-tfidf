
import csv
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import torch

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# Preprocessing function to clean the tweets
def preprocess_tweet(tweet):
    tweet = tweet.lower()  # convert to lowercase
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
def plot_tsne(embedding_matrix, labels):
    tsne = TSNE(n_components=2, perplexity=30, verbose=2)
    tsne_results = tsne.fit_transform(embedding_matrix)

    # Plot each cluster with a different color efficiently
    for i, label in enumerate(set(labels)):
        plt.scatter(tsne_results[labels == label, 0], tsne_results[labels == label, 1], label=label)
    plt.legend()
    plt.savefig('tsne.png')


# Load tweets from json
def load_tweets_from_json(json_file):
    with open(json_file, 'r') as f:
        json_ = json.load(f)

    tweets = [preprocess_tweet(tweet['full_text']) for tweet in json_]
    return tweets


def main():
    # model_tag = 'all-distilroberta-v1'
    model_tag = 'all-mpnet-base-v2 '
    tweets = load_tweets_from_json('postgres_public_feeds_entry.json')
    try:
        # Load npz file from disk
        file = np.load('tweet-embs-{}.npz'.format(model_tag))
        tweet_embs = file.f.arr_0
    except FileNotFoundError:
        # Convert list of tweets into list of length-32 lists of tweets
        batched_tweets = [tweets[i:i+32] for i in range(0, len(tweets), 32)]

        model = SentenceTransformer()
        tweet_embs = []
        # Compute intent embeddings
        with torch.no_grad():
            for batch in batched_tweets:
                tweet_embs.append(model.encode(batch))

        tweet_embs = np.concatenate(tweet_embs, 0)
        np.savez('tweet-embs-all-distilroberta-v1.npz', tweet_embs)

    # Kmeans clustering on the embeddings
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=1000).fit(tweet_embs)
    labels = kmeans.labels_

    # Separate the tweets into clusters
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(tweets[i])

    # Print 10 random tweets from each cluster
    for i, cluster in enumerate(clusters):
        print(f'\n\nCluster {i}:')
        for tweet in np.random.choice(cluster, n_clusters):
            print(tweet)

    ids = np.random.choice(np.arange(len(tweet_embs)), 2000)
    plot_tsne(tweet_embs[ids], labels[ids])

    print()


if __name__ == "__main__":
    main()
