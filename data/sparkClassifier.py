import pickle

import pyspark
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, StopWordsRemover
from pyspark.ml.feature import StringIndexer
import plotly.express as px
from pyspark.ml import Pipeline, PipelineModel
import time
from sklearn.metrics import roc_curve


class SparkClassifier:

    @staticmethod
    def reviews_analysis():
        filename = '/sparkTrainHistory'
        spark = SparkSession \
            .builder \
            .master('local') \
            .config('spark.mongodb.input.uri', 'mongodb://localhost:27017/DEDS.split_hotel_reviews') \
            .config('spark.mongodb.output.uri', 'mongodb://localhost:27017/DEDS.split_hotel_reviews') \
            .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.11:2.3.1') \
            .getOrCreate()

        df = spark.read \
            .format("com.mongodb.spark.sql.DefaultSource") \
            .option("database", "DEDS") \
            .option("collection", "split_hotel_reviews") \
            .load()
        df = df.drop("_id", "index")

        (train_set, val_set, test_set) = df.randomSplit([0.70, 0.15, 0.15], seed=2000)

        # Load training data
        start = time.time()
        tokenizer = Tokenizer(inputCol="review", outputCol="words")
        stopwords = StopWordsRemover(inputCol='words', outputCol='cleanedWords')
        cv = CountVectorizer(vocabSize=2 ** 16, inputCol="cleanedWords", outputCol='tf')
        idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms
        label_stringIdx = StringIndexer(inputCol="label", outputCol="value")
        lr = LogisticRegression(maxIter=100, fitIntercept=True, elasticNetParam=0.5)
        pipeline = Pipeline(stages=[tokenizer, stopwords, cv, idf, label_stringIdx, lr])

        model = PipelineModel.load("sparkLrModel")
        # model = pipeline.fit(train_set)
        end = time.time()
        # model.save("sparkLrModel")

        predictions = model.transform(val_set)
        accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())

        # Returns as a list (false positive rate, true positive rate)
        preds = predictions.select('label', 'probability') \
            .rdd.map(lambda row: (float(row['probability'][1]), float(row['label']))) \
            .collect()

        evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
        roc_auc = evaluator.evaluate(predictions)

        print("\nAmount of seconds for spark to finish model: %.2f" % (end - start))
        print("Accuracy Score: {0:.4f}".format(accuracy))
        print("ROC-AUC: {0:.4f}".format(roc_auc))

        # paramGrid = ParamGridBuilder() \
        #     .addGrid(lr.regParam, [0.1, 0.01]) \
        #     .addGrid(lr.fitIntercept, [False, True]) \
        #     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        #     .build()
        #
        # crossval = CrossValidator(estimator=pipeline,
        #                           estimatorParamMaps=paramGrid,
        #                           evaluator=BinaryClassificationEvaluator(),
        #                           numFolds=2)  # use 3+ folds in practice

        # cvModel = crossval.fit(train_set)
        # metrics = pickle.load(open(filename, 'rb'))

        # Pickle(save) metrics
        # pickle.dump(cvModel.avgMetrics, open(filename, 'wb'))

        # prediction = cvModel.transform(test_set)
        # selected = prediction.select("review", "label", "probability", "prediction")
        # print(selected.head(5))

        # accuracy figure

        y_score, y_true = zip(*preds)
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        spark_acc_fig = px.line(
            x=fpr,
            y=tpr, labels={"x": "False positive rate", "y": "True Positive rate"}, title='Model ROC', width=800)

        return spark_acc_fig
