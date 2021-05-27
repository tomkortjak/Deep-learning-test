import pandas as pd
from sqlalchemy import create_engine

from data.cleaner import CleaningMethods
from data.data_analyser import AnalyseMethods
from common.review_scrapers import ReviewScrapers
from data.db_handler import DBHandler
from data.sparkClassifier import SparkClassifier
from data.kerasClassifier import KerasClassifier
from common.dashboard import GraphPlot

if __name__ == '__main__':
    # importing data
    # booking_scraper = ReviewScrapers
    # hotel_reviews_csv = pd.read_csv('assets/csv/Hotel_Reviews.csv')
    # hotel_reviews_manual = pd.read_csv('assets/csv/Hotel_Review_Manual.csv')
    # hotel_reviews_booking_scraped = booking_scraper.scrapeBooking()
    # hotel_reviews_agoda_scraped = booking_scraper.scrapeAgoda()

    # hotel_reviews = pd.concat(
    #     [hotel_reviews_csv, hotel_reviews_manual, hotel_reviews_booking_scraped, hotel_reviews_agoda_scraped])
    # hotel_reviews.dropna(inplace=True)

    # Import static methods for database interaction
    db_handler = DBHandler()

    # Write all dataframes to the MySQL database
    # db_handler.set_mysql_database(hotel_reviews)

    # Get the reviews from the MySQL database
    # reviews = db_handler.get_mysql_database(max_rows=500000)

    # Transform and clean dataframe with column names
    # cleaner = CleaningMethods()
    # cleaner.prep_reviews(reviews)
    # split_reviews = cleaner.split_reviews(reviews)

    # Write reviews to NoSQL MongoDB
    # db_handler.set_nosql_database(reviews, split_reviews)

    # Read reviews from NoSQL MongoDB
    hotel_reviews = pd.DataFrame(db_handler.get_nosql_labeled_reviews())
    dashboard_df = pd.DataFrame(db_handler.get_nosql_full_reviews())
    review_count = pd.DataFrame(db_handler.nosql_mapreduce_query())
    review_count.rename(columns={"_id": "hotel", "value": "amount_reviews"}, inplace=True)

    # Cleanup and write to csv
    hotel_reviews.drop(['index', '_id'], inplace=True, axis=1)
    dashboard_df.drop(['_id'], inplace=True, axis=1)
    # hotel_reviews.to_csv('assets/csv/Hotel_Reviews.csv', index=False)

    hotel_reviews = hotel_reviews.sample(n=10000)
    # Spark
    spark = SparkClassifier()
    spark_acc_fig = spark.reviews_analysis()

    # Keras
    keras = KerasClassifier()
    keras_acc_fig, keras_loss_fig = keras.text_classification(hotel_reviews)

    # Removing duplicates so that the dashboard doesn't get overloaded
    dashboard_df.drop_duplicates('Hotel_Name', keep='first', inplace=True)

    # Dashboard
    dashboard = GraphPlot(dashboard_df, review_count, keras_acc_fig, keras_loss_fig, spark_acc_fig)
