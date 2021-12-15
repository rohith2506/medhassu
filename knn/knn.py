#!/usr/bin/env python3.9

import argparse
import logging
import numpy as np
import pandas as pd
import os

class DataManager:
    def __init__(self, data_location):
        self.data_folder = os.path.join(os.getcwd(), data_location)
        if not os.path.exists(self.data_folder):
            raise Exception("data folder {} does not exist".format(self.data_folder))

    def get_data(self):
        movies_df  = pd.read_csv(os.path.join(self.data_folder, "movies.csv"))
        ratings_df = pd.read_csv(os.path.join(self.data_folder, "ratings.csv"))

        logging.info("Total number of movies:  {}".format(len(movies_df)))
        logging.info("Total number of ratings: {}".format(len(ratings_df)))

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
        self.popular_threshold = 50

    def build_features(self):
        movies_ratings_df =  pd.merge(self.movies_df, self.ratings_df, on='movieId')

        # calculate movie and how many ratings it got
        movie_rating_count_df = movies_ratings_df.groupby(by = ['title'])['rating'].count().reset_index().rename(columns = {'rating': 'total_rating_count'})[['title', 'total_rating_count']]

        # merge with ratings
        ratings_with_total_rating_count = movies_ratings_df.merge(movie_rating_count_df, left_on='title', right_on='title', how='left')
        # Retrieve popular movies
        ratings_popular_movie_df = ratings_with_total_rating_count.query('total_rating_count >= @self.popular_threshold')
        # pivot the table
        movie_features_df = ratings_popular_movie_df.pivot_table(index = 'title', columns = 'userId', values = 'rating').fillna(0)

        return movie_features_df


    def recommend_movies(self, movie_name):
        pass


def setup_logging(args):
    if args.verbose:
        logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    else:
        logging.basicConfig(encoding='utf-8', level=logging.INFO)

def create_parser():
    parser = argparse.ArgumentParser(description='KNN algorithm implementation')

    parser.add_argument('--data-location', type=str, help='Data location to train and test the model', default='data')
    parser.add_argument('--log-level', type=str, help='Choices to choose from [DEBUG, INFO, WARNING, ERROR]', default='INFO')

    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    logging.basicConfig(encoding='utf-8', level=args.log_level)

    movie_recommender = MovieRecommender(args.data_location)
    movie_features_df = movie_recommender.build_features()
