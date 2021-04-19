import nltk
import random
import os.path
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
from nltk.tokenize import TweetTokenizer

stop_words = stopwords.words('english')

#lemmatizer = WordNetLemmatizer()
## Helper function that was used for cleaning data
# def __cleanData(tweet_tokens):
#     cleaned_tokens = ''
#     for token, tag in pos_tag(tweet_tokens):
#         token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
#         token = re.sub('(@[A-Za-z0-9_]+)', '', token)
#         if tag.startswith('NN'):
#             pos = 'n'
#         elif tag.startswith('VB'):
#             pos = 'v'
#         else:
#             pos = 'a'
#         token = lemmatizer.lemmatize(token, pos)
#         if len(token) > 0 and token not in string.punctuation:
#             cleaned_tokens += token.lower() + " "
#     return cleaned_tokens


class Classifier:
    def __init__(self, with_stopwords=False):
        self.train_data = []
        self.val_data = []
        self.model = None
        self.tfidf_vc = TfidfVectorizer(min_df = 10, max_features = 200000, analyzer = "word", ngram_range = (1, 2), lowercase = True, stop_words=stop_words)
        self.explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
        self.pickle_name = 'model_with_stopwords.pkl' if with_stopwords else 'model.pkl'
        self.tfidf_name = 'tfidf_vc_with_stopwords.pkl' if with_stopwords else 'tfidf_vc.pkl'

    def save_model(self):
        if self.model is None:
            print('Fit the model before saving it')
            return
        with open(self.pickle_name, 'wb') as file:
            print(f'Saving {self.pickle_name}...')
            pickle.dump(self.model, file)
            print(f"Saved.")
        with open(self.tfidf_name, 'wb') as file:
            print(f'Saving {self.tfidf_name}...')
            pickle.dump(self.tfidf_vc, file)
            print(f"Saved.")
        return self


    def load_data(self, test_size = 0.15):
        print('Loading data...')
        data_ = pd.read_csv('cleaned_data.csv')
        data = pd.DataFrame(data={
            'texts': data_.text.astype('U'),
            'labels': [0] * 800000 + [1] * 800000
            })
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=777)

        self.train_data = train_data
        self.val_data = val_data
        print('Data loaded')
        return self
    
    def fit(self):
        model_pickle_exists = os.path.isfile(self.pickle_name) 
        tfidf_pickle_exists = os.path.isfile(self.tfidf_name)

        if model_pickle_exists and tfidf_pickle_exists:
            with open(self.pickle_name, 'rb') as file:
                print(f'Using already trained model: {self.pickle_name} file')
                self.model = pickle.load(file)
            with open(self.tfidf_name, 'rb') as file:
                print(f'Using already tfidf model: {self.tfidf_name} file')
                self.tfidf_vc = pickle.load(file)
            return self
        else:
            print(f"Model pickle {self.pickle_name} exists - {model_pickle_exists}")
            print(f"Tfidf pickle {self.tfidf_name} exists - {tfidf_pickle_exists}")
                
        print('Fitting tfidf vector')
        train_vc = self.tfidf_vc.fit_transform(self.train_data.texts)
        print('Fitting done.')
        print('Training in progress...')
        self.model = LogisticRegression(C = 1.0, solver = "sag").fit(train_vc, self.train_data.labels)
        print('Training done.')
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


    def predict(self, text: str, output_to_html=False):
        pipeline = make_pipeline(self.tfidf_vc, self.model)

        print("Text: \n", text)
        print("Probability (Positive) =", f"{pipeline.predict_proba([text])[0, 1] * 100:.2f} %") 
        print("Probability (Negative) =", f"{pipeline.predict_proba([text])[0, 0] * 100:.2f} %")
        # print("True Class is:", class_names[self.val_data.labels[idx]])

        if output_to_html:
            with open("Output.html", "w") as text_file:
                explained_instance = self.explainer.explain_instance(text, pipeline.predict_proba, num_features = 10)
                text_file.write(explained_instance.as_html())

def main():
    model = Classifier(with_stopwords=False)                                # Instantiate the classifier
    model.load_data()                                    # Read data from file
    model.fit()                          # Fit logistic regression model with train data
    model.save_model()                                  # Save model to load later, instead of training all over again
    model.print_score()                                 # Print model f1 accuracy based on test data
    model.predict('i hate cats', output_to_html=True)   # Predict label on text


if __name__ == '__main__':
    main()