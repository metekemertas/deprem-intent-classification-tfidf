
import datasets
import json
import re
import numpy as np
import torch as th

from huggingface_hub import login
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          Trainer, TrainingArguments)


LABEL_NAMES = [
    'Alakasiz', 'Barinma', 'Elektronik',
    'Giysi', 'Kurtarma', 'Lojistik', 'Saglik',
    'Su', 'Yagma', 'Yemek']


class DepremTweetUnlabeledDataset(th.utils.data.Dataset):

    def __init__(self, tweets, tokenizer):
        self.tweets = tweets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        text = self.tweets[idx]
        encoding = self.tokenizer(text, max_length=64, padding="max_length", truncation=True)
        encoding = {key: th.tensor(val) for key, val in encoding.items()}
        encoding["labels"] = encoding["input_ids"]

        inp = encoding["input_ids"]
        tokens = range(len(inp))
        # We need to select 15% random tokens from the given list
        num_of_token_to_mask = round(len(tokens) * 0.15)
        token_to_mask = np.random.choice(np.array(tokens),
                                         size=num_of_token_to_mask,
                                         replace=False).tolist()
        # Now we have the indices where we need to mask the tokens
        inp[token_to_mask] = self.tokenizer.mask_token_id
        encoding["input_ids"] = inp

        return dict(encoding)


# Preprocessing function to clean the tweets
def preprocess_tweet(tweet):
    # remove handles, hashtags, urls
    # tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    # remove urls
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'www\S+', '', tweet)
    tweet = re.sub(r'pic.twitter\S+', '', tweet)

    tweet = re.sub(r'\W', ' ', tweet)  # remove special characters
    tweet = re.sub(r'\s+', ' ', tweet)  # remove multiple whitespaces

    return tweet.strip()


def prepare_datasets(json_path, tokenizer):
    with open(json_path, 'r') as f:
        json_ = json.load(f)

    tweets = [preprocess_tweet(tweet['full_text']) for tweet in json_]
    tweets = list(set(tweets))  # Remove duplicates
    print("Number of tweets: {}".format(len(tweets)))
    print(tweets[:10])
    print()

    n_train = int(len(tweets) * 0.85)
    train_ds = DepremTweetUnlabeledDataset(tweets[:n_train], tokenizer)
    val_ds = DepremTweetUnlabeledDataset(tweets[n_train:], tokenizer)

    return train_ds, val_ds


def main():
    login(token='<your-HF-token>')

    # model_name = "loodos/bert-base-turkish-uncased"
    model_name = "dbmdz/bert-base-turkish-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, val_ds = prepare_datasets(
        "postgres_public_feeds_entry.json", tokenizer)
    # Note: above code could be replaced with downloading the dataset from HF and preprocessing it
    # (see next line for example).
    # train_ds = datasets.load_dataset("deprem-private/deprem_tweet_unlabeled", "plain_text")
    # train_ds = clean_dataset(train_ds)

    model = AutoModelForMaskedLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir="./output-mlm",
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        num_train_epochs=1,
        eval_steps=1000,
        logging_steps=1000
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds
    )
    trainer.train()

    print()


if __name__ == '__main__':
    main()
