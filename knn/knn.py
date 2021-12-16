#!/usr/bin/env python3.9

import argparse
import logging
import numpy as np
import pandas as pd
from thefuzz import fuzz
import json
import os

class DataManager:
    def __init__(self, data_location):
        self.data_folder = os.path.join(os.getcwd(), data_location)
        if not os.path.exists(self.data_folder):
            raise Exception("data folder {} does not exist".format(self.data_folder))

    def get_data(self):
        movies_df  = pd.read_csv(os.path.join(self.data_folder, "movies.csv"))
        ratings_df = pd.read_csv(os.path.join(self.data_folder, "ratings.csv"))
        return movies_df, ratings_df


class KNN:
    def __init__(self, k):
        self.num_neighbours = k

    def calculate_euclidean_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def get_neighbours(self, train, test_row):
        distances = list()
        for train_row in train:
            distance = self.calculate_euclidean_distance(train_row, test_row)
            distances.append((train_row, distance))
        distance.sort(key = lambda tup: tup[1])
        neighbours = list()
        for i in range(self.num_neighbours):
            neighbours.append(distances[i][0])
        return neighbours

class MovieRecommender:
    def __init__(self, data_location):
        data_manager = DataManager(data_location)
        movies_df, ratings_df = data_manager.get_data()

        self.movies_df = movies_df
        self.ratings_df = ratings_df

        # This determines the popularity threshold. In this count, how many ratings it got
        self.popular_threshold = 20

        # Lets go with five nearest neighbours to begin with
        self.num_neighbours = 5
        self.knn = KNN(self.num_neighbours)


    def build_features(self):
        movies_ratings_df =  pd.merge(self.movies_df, self.ratings_df, on='movieId')

        # calculate movie and how many ratings it got
        movie_rating_count_df = movies_ratings_df.groupby(by = ['title'])['rating'].count().reset_index().rename(columns = {'rating': 'total_rating_count'})[['title', 'total_rating_count']]

        # merge with ratings
        ratings_with_total_rating_count = movies_ratings_df.merge(movie_rating_count_df, left_on='title', right_on='title', how='left')

        # Get popular movies
        rarings_with_popular_movies = ratings_with_total_rating_count.query('total_rating_count >= @self.popular_threshold')

        # pivot the table
        movie_features_df = ratings_with_total_rating_count.pivot_table(index = 'title', columns = 'userId', values = 'rating').fillna(0)

        return movie_features_df

    def calculate_similarity(self, movie_name_a, movie_name_b):
        return fuzz.ratio(movie_name_a, movie_name_b)

    def find_movie(self, movie_name, movie_features_df):
        maximum_similarity, matched_movie_name, matched_movie_series = None, None, None
        for idx in range(movie_features_df.shape[0]):
            movie = movie_features_df.iloc[idx].name
            similarity = self.calculate_similarity(movie, movie_name)
            if not maximum_similarity or similarity > maximum_similarity:
                maximum_similarity = similarity
                matched_movie_name = movie
                matched_movie_series = movie_features_df.iloc[idx]
        return maximum_similarity, matched_movie_name, matched_movie_series

    def recommend_movies(self, movie_name, movie_features_df):
        similarity, matched_movie_name, matched_movie_series = self.find_movie(movie_name, movie_features_df)
        logging.info("Matched movie name: {}\tSimilarity score: {}".format(matched_movie_name, similarity))

        if similarity <= 35:
            logging.error("Couldn't find better match in our hot list. Can't recommend for this exotic movie")

        neighbours = []
        for idx in range(movie_features_df.shape[0]):
            movie_series = movie_features_df.iloc[idx]
            distance = self.knn.calculate_euclidean_distance(movie_series.values, matched_movie_series.values)
            neighbours.append((distance, movie_features_df.iloc[idx].name))

        neighbours.sort()

        logging.info("\n\n\nHere are the top recommended movies")
        for distance, movie_name in neighbours[1:10]:
            logging.info("Movie name: {}\tdistance: {}".format(movie_name, distance))


def setup_logging(args):
    if args.verbose:
        logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    else:
        logging.basicConfig(encoding='utf-8', level=logging.INFO)

def create_parser():
    parser = argparse.ArgumentParser(description='Movie Recommender system using KNN algorithm')

    parser.add_argument('--data-location', type=str, help='Data location to train and test the model', default='data')
    parser.add_argument('--log-level', type=str, help='Choices to choose from [DEBUG, INFO, WARNING, ERROR]', default='INFO')
    parser.add_argument('--movie-name', type=str, help='Choose the movie name', required=True)

    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    logging.basicConfig(encoding='utf-8', level=args.log_level)

    movie_recommender = MovieRecommender(args.data_location)
    movie_features_df = movie_recommender.build_features()
    movie_recommender.recommend_movies(args.movie_name, movie_features_df)
