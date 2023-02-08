

import csv
import re
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer


def draw_plot(predictions):
    counts = Counter(predictions)
    plt.bar(["Kurtarma", "Yemek-Su", "Giysi"], [counts[0], counts[1], counts[2]])
    plt.ylabel("Tweet Count")
    plt.title("Tweet Count per Cluster Label")
    plt.savefig("intent_counts.png")


def get_most_common_words(strings, num_words):
    # Join the list of strings into one string
    all_text = " ".join(strings)

    # Split the string into words
    words = re.findall(r'\b\w+\b', all_text)

    # Count the frequency of each word
    word_counts = Counter(words)

    # Sort the words by frequency and return the top num_words
    return word_counts.most_common(num_words)


# Preprocessing function to clean the tweets
def preprocess_tweet(tweet):
    tweet = re.sub(r'\W', ' ', tweet) # remove special characters
    tweet = re.sub(r'\s+', ' ', tweet) # remove multiple whitespaces
    tweet = tweet.lower()  # convert to lowercase
    tweet = remove_diacritics(tweet)
    return tweet


def remove_diacritics(text):
    # define the mapping from diacritic characters to non-diacritic characters
    mapping = {
        '\u00c7': 'C', '\u00e7': 'c',
        '\u011e': 'G', '\u011f': 'g',
        '\u0130': 'I', '\u0131': 'i',
        '\u015e': 'S', '\u015f': 's',
        '\u00d6': 'O', '\u00f6': 'o',
        '\u00dc': 'U', '\u00fc': 'u',
        '\u0152': 'OE', '\u0153': 'oe',
        '\u0049': 'I', '\u0131': 'i',
    }

    # replace each diacritic character with its non-diacritic counterpart
    text = ''.join(mapping.get(c, c) for c in text)

    return text


# Define a function to perform intent classification based on TF-IDF
def classify_intent(tweets, intents, tfidf_matrix, vectorizer):
    # Transform the intents into vectors
    intent_vectors = vectorizer.transform(intents).toarray()
    # Calculate the cosine similarity between the tweets and the intents
    scores = np.matmul(tfidf_matrix.toarray(), intent_vectors.T)
    #  Get the index of the intent with the highest score
    intent_index = np.argmax(scores, -1)
    #  Set the intent index to -1 if the score is 0 meaning there was no match.
    intent_index[scores.sum(-1) <= 0] = -1

    return intent_index


def main():
    kurtarma_keywords = "ses geliyor sesi enkaz altinda bebek aylik cocuk annesi cocuklari aglayan aglama sesleri kurtarilmayi yardimedi mahsur yardim altinda kalanlar gocuk bina acil kat altindalar enkazaltindayim alinamiyor enkaz saatlerdir destek enkazda kurtarma talebi ulasilamayan"
    yemek_su_keywords = "gida cay talebi gida yemek aclik su corba susuzluk beslenme"
    giysi_keywords = "giysi giyim elbise talebi battaniye yagmurluk kazak corap soguk"

    # Load the tweets from the CSV file
    tweets = []
    with open('data_new.csv', 'r') as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader)
        for row in reader:
            tweets.append(preprocess_tweet(row[2]))
    # Print the most common words in the tweets
    pprint(get_most_common_words(tweets, 1000))

    intents = [kurtarma_keywords, yemek_su_keywords, giysi_keywords]
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer on the tweets and the intents
    tfidf_matrix = vectorizer.fit_transform(tweets + [kurtarma_keywords, yemek_su_keywords, giysi_keywords])
    # Perform intent classification using the existing code
    predictions = classify_intent(tweets, intents, tfidf_matrix[:-3], vectorizer)
    print("Number of tweets classified: {}/{}".format((predictions >= 0).sum(), len(tweets)))
    draw_plot(predictions)


if __name__ == '__main__':
    main()
