import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        self.locations = [
            'Albuquerque, New Mexico',
            'Carlsbad, California',
            'Chula Vista, California',
            'Colorado Springs, Colorado',
            'Denver, Colorado',
            'El Cajon, California',
            'El Paso, Texas',
            'Escondido, California',
            'Fresno, California',
            'La Mesa, California',
            'Las Vegas, Nevada',
            'Los Angeles, California',
            'Oceanside, California',
            'Phoenix, Arizona',
            'Sacramento, California',
            'Salt Lake City, Utah',
            'San Diego, California',
            'Tucson, Arizona'
        ]

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Parse the query string
            def is_valid_date(date_str: str) -> bool:
                """Check if the date string is in the correct format YYYY-MM-DD."""
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                    return True
                except ValueError:
                    return False
            query_params = parse_qs(environ.get("QUERY_STRING", ""))
            location = query_params.get("location", [None])[0]
            start_date_str = query_params.get("start_date", [None])[0]
            end_date_str = query_params.get("end_date", [None])[0]

            # Validate and convert start_date and end_date to datetime objects if they are provided
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str and is_valid_date(start_date_str) else None
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str and is_valid_date(end_date_str) else None

            if location:
                filtered_reviews = [
                review for review in reviews if
                (review['Location'] == location) and
                (start_date is None or datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date) and
                (end_date is None or datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date)
                ]
            else:
                filtered_reviews = [
                    review for review in reviews
                    if (review['Location'] in self.locations) and
                    (start_date is None or datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date) and
                    (end_date is None or datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date)
                ]


            for review in filtered_reviews:
                review_body = review['ReviewBody']
                sentiment = self.analyze_sentiment(review_body)
                review['sentiment'] = sentiment

            sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)
            # print(sorted_reviews)
            response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            except (ValueError):
                request_body_size = 0

            request_body = environ['wsgi.input'].read(request_body_size)

            if not request_body:
                response_body = json.dumps({"error": "Empty request body."}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            try:
                request_data = parse_qs(request_body.decode('utf-8'))
            except Exception as e:
                response_body = json.dumps({"error": "Invalid request format."}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            location = request_data.get("Location", [None])[0]
            review_body = request_data.get("ReviewBody", [None])[0]
            if location == "" or location not in self.locations:
                location = None
            if review_body == "":
                review_body = None
            
            if not location or not review_body:
                response_body = json.dumps({"error": "Location and ReviewBody are required."}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            review_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            new_review = {
                'ReviewId': review_id,
                'ReviewBody': review_body,
                'Location': location,
                'Timestamp': timestamp
            }

            reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()