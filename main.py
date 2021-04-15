import nltk
import pandas as pd
import re
import nltk
import string
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('twitter_samples')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = stopwords.words('english')


def cleanData(tokens):
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    for token, tag in pos_tag(tokens):
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

def main() -> None:
    tokensPositive = twitter_samples.tokenized('positive_tweets.json')
    tokensNegative = twitter_samples.tokenized('negative_tweets.json')

    print(tokensPositive[0])
    print(cleanData(tokens)(tokensPositive[0]))

    print(tokensNegative[0])
    print(cleanData(tokens)(tokensNegative[0]))


if __name__ == "__main__":
    main()