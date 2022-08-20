import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories datasets from the specified filepaths

    Args:
        messages_filepath: path to the messages dataset
        categories_filepath: path to the categories dataset

    Returns:
        df: merged pandas dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Cleans the merged dataset

    Args:
        df: pandas dataframe

    Returns:
        df: cleaned dataframe
    """
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each column value to be last character of the string
        categories[column] = categories[column].str[-1]
        # convert each column value to an integer
        categories[column] = categories[column].astype(int)
        # convert values to binary
        categories[column] = categories[column].clip(0,1)

    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)
    
    # drop NaN messages
    df.dropna(subset=["message"], axis=0, inplace=True) # drop the row
    
    # drop the id and original columns as they are not useful for the learning problem
    df.drop(["id", "original"], axis=1, inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Saves clean dataset into an sqlite database

    Args:
        df:  dataframe
        database_filename: name of the database file
    """

    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, if_exists="replace", index=False)


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
