import nltk
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

stopWords =  stopwords.words('english')

def removeStopWords(text: str) -> str:
    tokens = nltk.word_tokenize(text)
    return " ".join([token for token in tokens if token not in stopWords])

def readRawData(filePath: str ='./data.csv'):
    return pd.read_csv(filePath, index_col=0)["0"]

def cleanData(row):
    # TODO: remove links before this step
    matchAtUser = r"@\w+|#|:"
    cleaned1 = re.sub(matchAtUser, "",  row).strip()

    matchNonCharacters = r"(?!\s|\.|,)\W+"
    cleaned2 = re.sub(matchNonCharacters, '', cleaned1)

    matchDotsAndCommas = r"\.|,"
    cleaned3 = re.sub(matchDotsAndCommas, " ", cleaned2).lower()

    return cleaned3

def main() -> None:
    data = readRawData()

    for i in range(len(data)):
        cleaned = cleanData(data[i])
        withoutStopWords = removeStopWords(cleaned)
        print(i, withoutStopWords)

if __name__ == "__main__":
    main()