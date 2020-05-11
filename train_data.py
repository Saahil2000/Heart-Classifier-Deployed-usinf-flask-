import sys
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    """
        Function to load data from sqlite
        Input:
            - database_filepath: That path to the sqlite DB
        Output:
            - X: All independant variables
            - y: Target
            - List of columns

    """

    # Read data from sql to a DataFrame
    df = pd.read_sql_table("MLStaging", "sqlite:///"+database_filepath)

    # Separate X and y columns
    # X - independent
    # y - dependent
    X = df[["age", "sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]] #need to pass as an array if sending more than one value
    y = df["target"]

    print(y)

    return X, y, list(y)


def build_model():
    """
        Function to build the ML model
        Input:
            -
        Ouput:
            - GridSearchCV object
    """

    # Forming Pipleine
    pipeline = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('clf', RandomForestClassifier())
    ])

    # Initializing parameters for Grid search
    parameters = {
        'clf__n_estimators': [10, 50, 100]
    }

    # GridSearch Object with pipeline and parameters
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
        Function to evaluate model
    """

    # Predict results of X_test
    y_pred = model.predict(X_test)


    print(classification_report(Y_test, y_pred))
    print(accuracy_score(Y_test, y_pred))



def save_model(model, model_filepath):
    """
        Function to save the ML model
    """

    # open the file
    pickle_out = open(model_filepath, "wb")

    # write model to it
    pickle.dump(model, pickle_out)

    # close pickle file
    pickle_out.close()


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
        evaluate_model(model, X_test, Y_test)

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
