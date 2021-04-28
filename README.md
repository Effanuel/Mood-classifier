# Mood-classifier

### Natural text binary mood classifier

1. Clone

```zsh
git clone https://github.com/Effanuel/Mood-classifier.git
cd Mood-classifier
```

2. Init virtual environment

```zsh
virtualenv mood
source ./mood/bin/activate (or `.\mood\Scripts\activate`)
```

3. Intall dependencies

```zsh
pip install -r requirements.txt
```

4. Extract cleaned data
```zsh
tar xvf cleaned_data.csv.zip
```

5. Run

`python main.py` with line arguments:
```zsh
 Command line arguments:
            --print_score - prints f1 score, accuracy, confusion matrix of the model and opens confusion matrix heatmap
            --predict=<text> - predicts sentiment of a given text 
            --fit - loads data, fits logistic regresion model and saves the models to pickle files
            --stopwords - if fitting or predicting should be done on a model that used stopwords filtering      
            --open - saves the predicted label info in .html file and opens it in browser tab *(works only with --predict)*

        Examples: 
            * fit the model and show the score: `python main.py --fit --print_score`
            * fit model with stopwords: `python main.py --stopwords --fit`
            * predict sentiment of text and open info: `python main.py --open --predict="I love dogs")`
```
