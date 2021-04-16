import nltk
import pandas as pd
import random
import re
import nltk
import string
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.probability import ELEProbDist

# Uncomment to download required packages
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('twitter_samples')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

stop_words = stopwords.words('english')


def cleanData(tweet_tokens, stop_words = ()):
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def flatten_list(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":
    positive_cleaned_tokens_list = [cleanData(tokens, stop_words) for tokens in twitter_samples.tokenized('positive_tweets.json')] 
    negative_cleaned_tokens_list = [cleanData(tokens, stop_words) for tokens in twitter_samples.tokenized('negative_tweets.json')]

    all_pos_words = flatten_list(positive_cleaned_tokens_list)

    freq_dist_pos = nltk.FreqDist(all_pos_words)

    positive_dataset = [(tweet_dict, "Positive") for tweet_dict in get_tweets_for_model(positive_cleaned_tokens_list)]
    negative_dataset = [(tweet_dict, "Negative") for tweet_dict in get_tweets_for_model(negative_cleaned_tokens_list)]
    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)
    train_data = dataset[:8000]
    test_data = dataset[8000:]

    classifier = nltk.NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", nltk.classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    custom_tweet = "I hate cats so much xd"

    custom_tokens = cleanData(nltk.tokenize.word_tokenize(custom_tweet))

    feature_set = dict([token, True] for token in custom_tokens)
    print(custom_tweet, '->', classifier.classify(feature_set))