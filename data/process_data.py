import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories data from CSV files.

    Args:
        messages_filepath (str): Filepath to the messages CSV file.
        categories_filepath (str): Filepath to the categories CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing messages and categories.
    """
    try:
        # Load the messages and categories DataFrames from CSV files
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)

        # Merge the DataFrames based on the 'id' column
        df = pd.merge(messages, categories, on='id', how='left')

        return df
    except Exception as e:
        print('An error occurred')
        return None

def clean_data(df):
    """
    Clean and preprocess the DataFrame containing messages and categories.

    Args:
        df (pd.DataFrame): DataFrame containing merged messages and categories.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame with one column for each categorie.
    """
    try:
        # Split the 'categories' column on semicolons and expand it into separate binary columns
        categories = df['categories'].str.split(';', expand=True)
        
        # Extract category names by removing the last two characters
        row = categories.iloc[0]
        category_colnames = row.apply(lambda x: x[:-2])
        categories.columns = category_colnames
        
        # Convert values in category columns to integers
        for column in categories:
            categories[column] = categories[column].str[-1]
            categories[column] = categories[column].astype(int)

        # Convert to binary
        categories.replace(2, 1, inplace=True)
        
        # Drop the original 'categories' column
        df = df.drop('categories', axis=1)
        
        # Concatenate the processed category columns to the original DataFrame
        df = pd.concat([df, categories], axis=1)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove rows with missing values
        df = df.dropna()
        
        return df
    except Exception as e:
        print("An error occurred")
        return None

def save_data(df, database_filename):
    """
    Save the cleaned data to an SQLite database.

    Args:
        df (pd.DataFrame): Cleaned and preprocessed DataFrame.
        database_filename (str): Filepath for the SQLite database.

    Returns:
        None
    """
    try:
        engine = create_engine('sqlite:///{}'.format(database_filename))
        df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
        print('Data successfully saved to{}'.format(database_filename))
    except Exception as e:
        print("An error occurred:")

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
