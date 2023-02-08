
import csv
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import torch

from intent_classification_xlm import *
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# Plot t-SNE visualization of the embedding matrix
def plot_tsne(embedding_matrix, labels):
    tsne = TSNE(n_components=2, perplexity=30, verbose=2)
    tsne_results = tsne.fit_transform(embedding_matrix)

    plt.figure(figsize=(16, 10))
    for i in range(len(labels)):
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1], label=labels[i])
    plt.legend()
    plt.show()


# Load tweets from json
def load_tweets_from_json(json_file):
    with open(json_file, 'r') as f:
        json_ = json.load(f)

    tweets = [preprocess_tweet(tweet['full_text']) for tweet in json_]
    return tweets


def main():
    try:
        tweets = load_tweets_from_json('postgres_public_feeds_entry.json')
        # Load npz file from disk
        file = np.load('tweet_embs.npz')
        tweet_embs = file.f.arr_0
    except FileNotFoundError:
        tweets = load_tweets_from_json('postgres_public_feeds_entry.json')
        # Convert list of tweets into list of length-32 lists of tweets
        batched_tweets = [tweets[i:i + 32] for i in range(0, len(tweets), 32)]

        model = SentenceTransformer('all-distilroberta-v1')
        tweet_embs = []
        # Compute intent embeddings
        with torch.no_grad():
            for batch in batched_tweets:
                tweet_embs.append(model.encode(batch))

        tweet_embs = np.concatenate(tweet_embs, 0)
        np.savez('tweet_embs.npz', tweet_embs)

    # Kmeans clustering on the embeddings
    n_clusters = 20
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=1000).fit(tweet_embs)
    labels = kmeans.labels_

    # Separate the tweets into clusters
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(tweets[i])

    # Print 10 random tweets in each cluster
    for i, cluster in enumerate(clusters):
        print(f'\n\nCluster {i}:')
        for tweet in np.random.choice(cluster, 10):
            print(tweet)

    plot_tsne(tweet_embs, labels)

    print()


if __name__ == "__main__":
    main()
