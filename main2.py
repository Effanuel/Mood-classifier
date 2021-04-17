import nltk
import random
import re
import string
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.probability import ELEProbDist
from lime.lime_text import LimeTextExplainer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
from sklearn.metrics import f1_score
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt

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
    return " ".join(cleaned_tokens)


class Classifier:
    def __init__(self):
        self.train_data = []
        self.val_data = []
        self.model = None
        self.tfidf_vc = TfidfVectorizer(min_df = 10, max_features = 10000, analyzer = "word", ngram_range = (1, 2), lowercase = True)


    def loadData(self, test_size = 0.15):
        positive_cleaned_tokens_list = [cleanData(tokens, stop_words) for tokens in twitter_samples.tokenized('positive_tweets.json')]
        negative_cleaned_tokens_list = [cleanData(tokens, stop_words) for tokens in twitter_samples.tokenized('negative_tweets.json')]
        data = pd.DataFrame(data={
            'texts': positive_cleaned_tokens_list + negative_cleaned_tokens_list,
            'labels': [1] * len(positive_cleaned_tokens_list) + [0] * len(negative_cleaned_tokens_list)
            })
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=777)

        self.train_data = train_data
        self.val_data = val_data
        return self
    
    def fit(self):
        train_vc = self.tfidf_vc.fit_transform(self.train_data.texts)
        self.model = LogisticRegression(C = 0.5, solver = "sag").fit(train_vc, self.train_data.labels)
        return self

    def print_score(self):
        if self.model is None:
            print('Fit the model before predicting labels')
            return

        val_vc = self.tfidf_vc.transform(self.val_data.texts)
        val_pred = self.model.predict(val_vc)
        val_cv = f1_score(self.val_data.labels, val_pred, average = "binary")
        print(val_cv)
        return self


    def predict(self, index=0, output_to_html=False):
        idx = self.val_data.index[index]

        pipeline = make_pipeline(self.tfidf_vc, self.model)
        class_names = ["Negative", "Positive"]
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(self.val_data.texts[idx], pipeline.predict_proba, num_features = 10)

        print("Text: \n", self.val_data.texts[idx])
        print("Probability (Positive) =", pipeline.predict_proba([self.val_data.texts[idx]])[0, 1])
        print("Probability (Negative) =", pipeline.predict_proba([self.val_data.texts[idx]])[0, 0])
        print("True Class is:", class_names[self.val_data.labels[idx]])
        print(exp.as_list())

        if output_to_html:
            with open("Output.html", "w") as text_file:
                text_file.write(exp.as_html())



def main():
    model = Classifier()
    model = model.loadData().fit().print_score().predict(2, output_to_html=True)


if __name__ == '__main__':
    main()