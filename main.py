import os.path
import re
import webbrowser
import getopt
import sys
import seaborn as sn
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from lime.lime_text import LimeTextExplainer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix

def show_heatmap(conf_matrix):
    ''' Plots confusion matrix heatmap '''
    confusion_matrix_dataframe = pd.DataFrame(conf_matrix, ['Positive', 'Negative'], columns=['Positive', 'Negative'])
    sn.set(font_scale=1.2)
    sn.heatmap(confusion_matrix_dataframe, annot=True, annot_kws={'size': 14}, fmt='.2%', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.xlabel('Predicted label') 
    plt.ylabel('True label') 
    plt.show()

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
        '''
        Saves the model to pickle format, so they can be loaded later, without the need for training
        This classifier currently supports a model with stopwords used and model without stopwords
        '''
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
        '''
        Reads and loads the data into the classifier
        '''
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
        '''
        Fits the LogisticRegression and tfidf models, if pickles are not used
        '''
        if self.model is not None:
            print('Model has already been fit.')
            return
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
        '''
        Prints f1 score, accuracy, confusion matrix, and opens confusion matrix heatmap
        '''
        if self.model is None:
            print('Fit the model before predicting labels')
            return

        predicted_labels = self.model.predict(self.tfidf_vc.transform(self.val_data.texts))

        accuracy_percentage = (1 - (self.val_data.labels != predicted_labels).sum()/float(predicted_labels.size)) * 100
        print(f'\nAccuracy: {accuracy_percentage:.2f} %')
        val_cv = f1_score(self.val_data.labels, predicted_labels, average = "binary")
        print(f'f1 score: {val_cv}')
        conf_matrix = confusion_matrix(self.val_data.labels, predicted_labels, normalize='true')
        print('Confusion matrix:\n', conf_matrix)
        show_heatmap(conf_matrix)
        return self


    def predict(self, text: str, open_output_file=False):
        '''
        Predicts text sentiment (Positive or Negative)
        ''' 
        pipeline = make_pipeline(self.tfidf_vc, self.model)

        print("Text: ", text)
        print("Probability (Positive) =", f"{pipeline.predict_proba([text])[0, 1] * 100:.2f} %") 
        print("Probability (Negative) =", f"{pipeline.predict_proba([text])[0, 0] * 100:.2f} %")

        if open_output_file:
            with open("Output.html", "w") as text_file:
                explained_instance = self.explainer.explain_instance(text, pipeline.predict_proba, num_features = 10)
                text_file.write(explained_instance.as_html())
            if os.path.isfile('Output.html'):
                print('Opening webrowser to display output')
                webbrowser.open(f"file://{os.path.abspath('Output.html')}", new=2)
            else:
                print('Output.html file cannot be found.')

def main():
    if (len(sys.argv) < 2):
        print('''
        Command line arguments:
            --print_score - prints f1 score, accuracy, confusion matrix of the model and opens confusion matrix heatmap
            --predict=<text> - predicts sentiment of a given text 
            --fit - loads data, fits logistic regresion model and saves the models to pickle files
            --stopwords - if fitting or predicting should be done on a model that used stopwords filtering      
            --open - saves the predicted label info in .html file and opens it in browser tab *(works only with --predict)*

        Examples: 
            * fit the model and show the score: `python main.py --fit --print_score`
            * predict sentiment of text and open info: `python main.py --open --predict="I love dogs")`
        ''')

    model = Classifier()
    open_output_file = False

    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'p:fso', ['predict=', 'fit', 'stopwords', 'open', 'print_score'])
        for opt, arg in options:
            if opt in ('--open'):
                open_output_file = True # opens output file in browser
            elif opt in ('-s', '--stopwords'):
                model.pickle_name = 'model_with_stopwords.pkl'
                model.tfidf_name = 'tfidf_vc_with_stopwords.pkl'
            elif opt in ('-f', '--fit'):
                model.load_data() # Read data from file
                model.fit() # Fit logistic regression model with train data
                model.save_model() # Save model to load later, instead of training all over again
            elif opt in ('-p', '--predict'):
                model.fit() # Fit logistic regression model with train data
                model.predict(arg, open_output_file=open_output_file)    # Predict label on text
            elif opt in ('--print_score'):
                model.load_data()
                model.fit() # Fit logistic regression model with train data
                model.print_score() # prints accuracy and f1 score
    except:
        print('Error with command line arguments')

if __name__ == '__main__':
    main()