import pandas as pd
from bson import Code
from sqlalchemy import create_engine
from pymongo import MongoClient


class DBHandler:
    """
    Class to perform basic functions on the local instance of the MySQL database.
    """

    @staticmethod
    def set_mysql_database(reviews):
        """
        Write to local MySQL database.
        """
        engine = create_engine('mysql://admin:adminadmin123@localhost:3306/reviews')
        reviews.to_sql(name='reviews', con=engine, if_exists='replace', index=False)

    @staticmethod
    def get_mysql_database(max_rows):
        """
        Read out the MySQL database with a parameter.
        :return: List containing all of the rows returned by the database
        """
        engine = create_engine('mysql://admin:adminadmin123@localhost:3306/reviews')
        connection = engine.raw_connection()
        cursor = connection.cursor()
        cursor.callproc("GetAllReviews", [max_rows])
        results = pd.DataFrame(cursor.fetchall())
        cursor.close()
        return results

    @staticmethod
    def set_nosql_database(reviews, split_reviews):
        """
        Write to local NoSQL mongoDB.
        """
        client = MongoClient("mongodb://localhost:27017/")
        db = client['DEDS']

        # insert reviews
        collection = db['hotel_reviews']
        reviews.reset_index(inplace=True)
        data_dict = reviews.to_dict("records")
        collection.insert_many(data_dict)

        # insert split reviews
        collection = db['split_hotel_reviews']
        split_reviews.reset_index(inplace=True)
        data_dict = split_reviews.to_dict("records")
        collection.insert_many(data_dict)

    @staticmethod
    def get_nosql_full_reviews():
        """
        Read From NoSQL mongoDB.
        """
        client = MongoClient("mongodb://localhost:27017/")
        db = client['DEDS']
        col = db["hotel_reviews"]

        query = {}
        query["$and"] = [
            {
                u"review": {
                    u"$ne": u""
                }
            },
            {
                u"review": {
                    u"$ne": u"nan"
                }
            }
        ]
        docs = list(col.find(query))
        client.close()
        return docs

    @staticmethod
    def get_nosql_labeled_reviews():
        """
        Read From NoSQL mongoDB.
        """
        client = MongoClient("mongodb://localhost:27017/")
        db = client['DEDS']
        col = db["split_hotel_reviews"]

        query = {}
        query["$and"] = [
            {
                u"review": {
                    u"$ne": u""
                }
            },
            {
                u"review": {
                    u"$ne": u"nan"
                }
            }
        ]
        docs = list(col.find(query))
        client.close()
        return docs

    @staticmethod
    def nosql_mapreduce_query():
        client = MongoClient("mongodb://localhost:27017/")
        db = client['DEDS']
        col = db["hotel_reviews"]

        try:
            mapping = Code("""
                            function () {
                                emit(this.Hotel_Name, 1);
                            };
                            """)
            reduce = Code("""
                                    function (key, values) {
                                        var reducedValue = "" + Array.sum(values);
                                        return reducedValue;
                                    };
                                    """)
            results = col.map_reduce(mapping, reduce, 'map_reduce_result').find()
            client.close()
            return (list(results))
        except Exception as e:
            raise Exception(e)
