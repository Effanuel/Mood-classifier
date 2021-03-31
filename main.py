import nltk
import pandas as pd
import re

def readRawData(filePath: str ='./data.csv'):
    return pd.read_csv(filePath, index_col=0)["0"]

def cleanData(row):
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
        print(i, cleanData(data[i]))

if __name__ == "__main__":
    main()