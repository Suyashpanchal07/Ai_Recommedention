import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time
from requests.exceptions import RequestException, HTTPError, ConnectionError

# Constants
TMDB_API_KEY = 'ccbf4ac32b70a26764e96f0b223c6aaa'  # Replace with your TMDb API key
GOOGLE_BOOKS_API_KEY = 'AIzaSyClmOg-zlJ_UZMOoirCGxNzNgxL318l5jo'  # Replace with your Google Books API key
TMDB_API_URL = 'https://api.themoviedb.org/3/search/movie'
TMDB_SIMILAR_MOVIES_URL = 'https://api.themoviedb.org/3/movie/{movie_id}/similar'
GOOGLE_BOOKS_API_URL = 'https://www.googleapis.com/books/v1/volumes'

# Sample Dataset File Names
MOVIE_DATA_FILE = 'movies.csv'
BOOK_DATA_FILE = 'books.csv'

# Fetch details from the TMDb API
def fetch_movie_details(movie_name):
    url = f'{TMDB_API_URL}?api_key={TMDB_API_KEY}&query={movie_name}'
    retries = 3
    while retries > 0:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            if results:
                movie = results[0]
                return {
                    'id': movie.get('id'),
                    'title': movie.get('title'),
                    'overview': movie.get('overview'),
                    'release_date': movie.get('release_date'),
                    'rating': movie.get('vote_average')
                }
            else:
                print("No results found for the movie.")
                return None
        except (RequestException, HTTPError, ConnectionError) as e:
            print(f"Request failed: {e}")
            retries -= 1
            time.sleep(2)
    print("Failed to fetch movie details after several retries.")
    return None

def fetch_similar_movies_from_api(movie_id):
    url = TMDB_SIMILAR_MOVIES_URL.format(movie_id=movie_id) + f'?api_key={TMDB_API_KEY}'
    retries = 3
    while retries > 0:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            similar_movies = []
            for movie in results:
                similar_movies.append({
                    'title': movie.get('title'),
                    'overview': movie.get('overview'),
                    'release_date': movie.get('release_date'),
                    'rating': movie.get('vote_average')
                })
            return similar_movies
        except (RequestException, HTTPError, ConnectionError) as e:
            print(f"Request failed: {e}")
            retries -= 1
            time.sleep(2)
    print("Failed to fetch similar movies after several retries.")
    return []

def fetch_book_details(book_name):
    url = f'{GOOGLE_BOOKS_API_URL}?q={book_name}&key={GOOGLE_BOOKS_API_KEY}'
    retries = 3
    while retries > 0:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            items = data.get('items', [])
            if items:
                book = items[0]['volumeInfo']
                return {
                    'title': book.get('title'),
                    'authors': ', '.join(book.get('authors', [])),
                    'description': book.get('description'),
                    'average_rating': book.get('averageRating', 'N/A')
                }
            else:
                print("No results found for the book.")
                return None
        except (RequestException, HTTPError, ConnectionError) as e:
            print(f"Request failed: {e}")
            retries -= 1
            time.sleep(2)
    print("Failed to fetch book details after several retries.")
    return None

def fetch_similar_books_from_api(book_name):
    url = f'{GOOGLE_BOOKS_API_URL}?q={book_name}&key={GOOGLE_BOOKS_API_KEY}'
    retries = 3
    while retries > 0:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            items = data.get('items', [])
            similar_books = []
            for item in items:
                book = item['volumeInfo']
                similar_books.append({
                    'title': book.get('title'),
                    'authors': ', '.join(book.get('authors', [])),
                    'description': book.get('description'),
                    'average_rating': book.get('averageRating', 'N/A')
                })
            return similar_books
        except (RequestException, HTTPError, ConnectionError) as e:
            print(f"Request failed: {e}")
            retries -= 1
            time.sleep(2)
    print("Failed to fetch similar books after several retries.")
    return []

def recommend_items_from_dataset(item_name, num_recommendations=5, item_type='movie'):
    try:
        dataset_files = {
            'movie': MOVIE_DATA_FILE,
            'book': BOOK_DATA_FILE
        }

        data = pd.read_csv(dataset_files[item_type])
        if item_type == 'movie':
            data.set_index('Title', inplace=True)
            if item_name not in data.index:
                print(f"Movie '{item_name}' not found in the dataset.")
                return []

            data['features'] = data['Genre'] + ' ' + data['Overview']
        elif item_type == 'book':
            data.set_index('Title', inplace=True)
            if item_name not in data.index:
                print(f"Book '{item_name}' not found in the dataset.")
                return []

            data['features'] = data['Genre'] + ' ' + data['Description']

        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['features'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        cosine_sim_df = pd.DataFrame(cosine_sim, index=data.index, columns=data.index)
        
        similar_items = cosine_sim_df[item_name].sort_values(ascending=False)[1:num_recommendations+1]
        return similar_items
    except Exception as e:
        print(f"Error processing recommendations: {e}")
        return []

def main():
    item_type = input("What would you like to search for (movie, book)? ").strip().lower()
    item_name = input(f"Enter the {item_type} name: ").strip()

    if item_type == 'movie':
        details = fetch_movie_details(item_name)
        if details:
            movie_id = details['id']
            dataset_recommendations = recommend_items_from_dataset(item_name, item_type='movie')
            if not dataset_recommendations:
                print(f"\nFetching similar movies from API since '{item_name}' is not in the dataset.")
                api_recommendations = fetch_similar_movies_from_api(movie_id)
                if api_recommendations:
                    print(f"\nSimilar movies to '{item_name}':")
                    for movie in api_recommendations:
                        print(f"Title: {movie['title']}")
                        print(f"Overview: {movie['overview']}")
                        print(f"Release Date: {movie['release_date']}")
                        print(f"Rating: {movie['rating']}")
                        print("-" * 40)
                else:
                    print(f"No recommendations found for Movie '{item_name}'.")
            else:
                print(f"\nRecommendations similar to '{item_name}' from dataset:")
                for item, score in dataset_recommendations.items():
                    print(f"{item}: Similarity Score {score:.2f}")

    elif item_type == 'book':
        details = fetch_book_details(item_name)
        if details:
            print(f"\nDetails for Book '{item_name}':")
            for key, value in details.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            dataset_recommendations = recommend_items_from_dataset(item_name, item_type='book')
            if not dataset_recommendations:
                print(f"\nFetching similar books from API since '{item_name}' is not in the dataset.")
                api_recommendations = fetch_similar_books_from_api(item_name)
                if api_recommendations:
                    print(f"\nSimilar books to '{item_name}':")
                    for book in api_recommendations:
                        print(f"Title: {book['title']}")
                        print(f"Authors: {book['authors']}")
                        print(f"Description: {book['description']}")
                        print(f"Average Rating: {book['average_rating']}")
                        print("-" * 40)
                else:
                    print(f"No recommendations found for Book '{item_name}'.")
            else:
                print(f"\nRecommendations similar to '{item_name}' from dataset:")
                for item, score in dataset_recommendations.items():
                    print(f"{item}: Similarity Score {score:.2f}")

    else:
        print(f"Invalid item type '{item_type}'. Valid types are: movie, book.")
        return

if __name__ == "__main__":
    main()
