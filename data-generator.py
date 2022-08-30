import pandas as pd
from random import randint


def generate_data(n_books=3000, n_genres=10, n_authors=450, n_publishers=50, n_readers=30000, dataset_size=100000):
    '''
    This function will generate a dataset with features associated to
    book data set. The dataset will have the following columns : 
        - book_id (String) : Unique identified for the book
        - book_rating (Integer) : A value between 0 and 10
        - reader_id (String) : Unique identifier for the user
        - book_genre (Integer) : An integer representing a genre for the book, 
                                 value is between 1 and 15, indicating that 
                                 there are 15 unique genres. Each book can only
                                 have 1 genre
        - author_id (String) : Unique identifier for the author of the book
        - num_pages (Integer) : Random value between 70 and 500
        - publisher_id (String) : A unique identifier for the publisher of the book
        - publish_year (Integer) : The year of book publishing
        - book_price (Integer) : The sale price of the book
        - text_lang (Integer) : The language of the book - returns an integer which 
                                is mapped to some language

    params:
        n_books (Integer) : The number of books you want the dataset to have
        n_genres (Integer) : Number of genres to be chosen from
        n_authors (Integer) : Number of authors to be generated
        n_publishers (Integer) : Number of publishers for the dataset
        n_readers (Integer) : Number of readers for the dataset
        dataset_size (Integer) : The number of rows to be generated 

    example:
        data = generate_data()
    '''

    d = pd.DataFrame(
        {
            'book_id': [randint(1, n_books) for _ in range(dataset_size)],
            'author_id': [randint(1, n_authors) for _ in range(dataset_size)],
            'book_genre': [randint(1, n_genres) for _ in range(dataset_size)],
            'reader_id': [randint(1, n_readers) for _ in range(dataset_size)],
            'num_pages': [randint(75, 700) for _ in range(dataset_size)],
            'book_rating': [randint(1, 10) for _ in range(dataset_size)],
            'publisher_id': [randint(1, n_publishers) for _ in range(dataset_size)],
            'publish_year': [randint(2000, 2021) for _ in range(dataset_size)],
            'book_price': [randint(1, 200) for _ in range(dataset_size)],
            'text_lang': [randint(1, 7) for _ in range(dataset_size)]
        }
    ).drop_duplicates()
    return d


d = generate_data(dataset_size=100000)
d.to_csv('data.csv', index=False)
