import joblib
import re
import sys

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
from sqlalchemy import create_engine
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


PUNCTUATION_REGEX = re.compile(r"[^\w\s]")
STOPWORDS = stopwords.words('english')
WORDNET_LEMMATIZER = WordNetLemmatizer()
POS_TAGS_TO_LEMMATIZE = ["n", "v"]

def load_data(database_filepath):
    """
    Loads data from database

    Args:
        database_filepath: path to database

    Returns:
        (DataFrame) X: feature
        (DataFrame) Y: labels

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_query('select * from messages', engine)

    X = df['message'].values
    Y = df.drop(columns=['message','genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes a given text.

    Args:
        text: text string

    Returns:
        tokens: list of tokens

    """
    
    # lowercase string and remove punctuation
    text = PUNCTUATION_REGEX.sub(" ", text.lower()).strip()
    # tokenize text
    tokens = [token for token in word_tokenize(text)]
    # lemmatize text based on pos tags
    for pos_tag in POS_TAGS_TO_LEMMATIZE:
        tokens = [WORDNET_LEMMATIZER.lemmatize(token, pos=pos_tag) for token in word_tokenize(text)]
    # remove stopwords
    tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def build_model():
    """Builds classification model """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier()))
    ])

    parameters = {
        "clf__estimator__max_depth": [4, 8, 16],
        "clf__estimator__colsample_bytree":[0.5, 0.75, 1],
        "clf__estimator__learning_rate":[0.1,]
    }

    cv = GridSearchCV(pipeline, cv=3, param_grid=parameters, verbose=3, n_jobs=-1, scoring="f1_micro")
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model against a test dataset

    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        category_names: String array of category names
    """
    y_preds = model.predict(X_test)
    print(classification_report(y_preds, Y_test.values, target_names=category_names))
    # collect accuracy scores in a dict
    category_name_2_accuracy_score = {}
    for i in range(len(category_names)):
        category_name_2_accuracy_score[Y_test.columns[i]] = accuracy_score(Y_test.values[:,i],y_preds[:,i])
    print("Accuracy per category")
    print(pd.Series(category_name_2_accuracy_score))


def save_model(model, model_filepath):
    """
    Save the model to a Python pickle

    Args:s
        model: Trained model
        model_filepath: Path where to save the model
    """
    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
