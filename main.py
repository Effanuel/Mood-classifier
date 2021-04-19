import nltk
import random
import os.path
import re
import webbrowser
import getopt
import sys
from nltk.corpus import stopwords
from lime.lime_text import LimeTextExplainer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

# stop_words = stopwords.words('english')

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
    def __init__(self):
        self.train_data = []
        self.val_data = []
        self.model = None
        self.tfidf_vc = TfidfVectorizer(min_df = 10, max_features = 200000, analyzer = "word", ngram_range = (1, 2), lowercase = True)
        self.explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
        self.pickle_name = 'model.pkl'
        self.tfidf_name = 'tfidf_vc.pkl'

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
        if len(self.train_data) != 0 and len(self.val_data) != 0:
            print('Data is already loaded.')
            return self
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

        val_pred = self.model.predict(self.tfidf_vc.transform(self.val_data.texts))

        accuracy_percentage = (1 - (self.val_data.labels != val_pred).sum()/float(val_pred.size)) * 100
        print(f'\nAccuracy: {accuracy_percentage:.2f} %')
        val_cv = f1_score(self.val_data.labels, val_pred, average = "binary")
        print(f'f1 score: {val_cv}')
        return self


    def predict(self, text: str, output_to_html=False, open_output_file=False):
        pipeline = make_pipeline(self.tfidf_vc, self.model)

        print("Text: ", text)
        print("Probability (Positive) =", f"{pipeline.predict_proba([text])[0, 1] * 100:.2f} %") 
        print("Probability (Negative) =", f"{pipeline.predict_proba([text])[0, 0] * 100:.2f} %")
        # print("True Class is:", class_names[self.val_data.labels[idx]])

        if output_to_html:
            with open("Output.html", "w") as text_file:
                explained_instance = self.explainer.explain_instance(text, pipeline.predict_proba, num_features = 10)
                text_file.write(explained_instance.as_html())

        if open_output_file and output_to_html:
            if os.path.isfile('Output.html'):
                print('Opening webrowser to display output')
                webbrowser.open(f"file://{os.path.abspath('Output.html')}", new=2)
            else:
                print('Output.html file cannot be found.')

def main():
    model = Classifier()
    output_to_html = False
    open_output_file = False

    options, remainder = getopt.getopt(sys.argv[1:], 'p:fos', ['predict=', 'fit', 'output', 'stopwords', 'open', 'print_score'])
    for opt, arg in options:
        if opt in ('--open'):
            open_output_file = True # opens output file in browser
        elif opt in ('-o', '--output'):
            output_to_html = True # output results to html file
        elif opt in ('-s', '--stopwords'):
            model.pickle_name = 'model_with_stopwords.pkl' 
            model.tfidf_name = 'tfidf_vc_with_stopwords.pkl'
        elif opt in ('-f', '--fit'):
            model.load_data() # Read data from file
            model.fit() # Fit logistic regression model with train data
            model.save_model() # Save model to load later, instead of training all over again
        elif opt in ('-p', '--predict'):
            model.fit() # Fit logistic regression model with train data
            model.predict(arg, output_to_html=output_to_html, open_output_file=open_output_file)    # Predict label on text
        elif opt in ('--print_score'):
            model.load_data()
            model.print_score() # prints accuracy and f1 score

if __name__ == '__main__':
    main()