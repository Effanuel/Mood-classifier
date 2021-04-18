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

4. Run

```zsh
python main.py
```


Data downloaded from [here](https://www.kaggle.com/kazanova/sentiment140), tokenized with TweetTokenizer and cleaned by removing links and @'s

* https://towardsdatascience.com/what-makes-your-question-insincere-in-quora-26ee7658b010
* ... might be helpful https://realpython.com/python-nltk-sentiment-analysis/
