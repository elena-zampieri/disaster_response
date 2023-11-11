import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
import string

def load_data(database_filepath):
    """
    Load and prepare data from an SQLite database.

    Args:
        database_filepath (str): Filepath to the SQLite database.

    Returns:
        tuple: A tuple containing feature (X), target (Y), and category names.
    """
    try:
        engine = create_engine('sqlite:///{}'.format(database_filepath))
        df = pd.read_sql_table('DisasterResponse', con=engine)

        # Extract the feature (X) and target (Y) data
        X = df['message']
        Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
        
        # Get the category names
        category_names = Y.columns.tolist()

        return X, Y, category_names
    except Exception as e:
        print('An error occurred:{}'.format(str(e)))
        return None, None, None  # Return None values to indicate an error

def tokenize(text):
    """
    Tokenize and preprocess text data.

    Args:
        text (str): Input text to be tokenized.

    Returns:
        list: List of preprocessed tokens.
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    return tokens

def build_model():
    """
    Build a machine learning model pipeline.

    Returns:
        sklearn.pipeline.Pipeline: Model pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the machine learning model.

    Args:
        model: Machine learning model to be evaluated.
        X_test: Test features.
        Y_test: True test labels.
        category_names: List of category names.

    Returns:
        None
    """
    Y_pred = model.predict(X_test)

    for i, category_name in enumerate(category_names):
        print('Category:{}'.format(category_name))
        print(classification_report(Y_test[category_name], Y_pred[:, i]))
        print("\n" + "=" * 60 + "\n")

def save_model(model, model_filepath):
    """
    Save the trained machine learning model to a file.

    Args:
        model: Trained machine learning model.
        model_filepath: Filepath to save the model.

    Returns:
        None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

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
