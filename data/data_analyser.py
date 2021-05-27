import matplotlib.pyplot as plt
from langdetect import detect_langs
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, recall_score, precision_score, \
    plot_precision_recall_curve, average_precision_score
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression


class AnalyseMethods:
    def __init__(self):
        # Create stopwords, these are based on common words we have found in the reviews that hold no value to the
        # sentiment analysis.
        self.stopwords = set(STOPWORDS)
        self.stopwords.update(
            ["night", "coming", "many", "lots", "seperate", "basket", "duvets", "hotel", "iron", "channels",
             'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're',
             'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn', 'room', 'rooms', 'great', 'nice'])

    def show_wordcloud(self, df, label):
        my_cloud = WordCloud(
            background_color='white',
            stopwords=self.stopwords,
            random_state=42
        ).generate(' '.join(df['review']))
        plt.imshow(my_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(label, fontsize=24)
        plt.show()

    def detect_langs(self, df, max_rows):
        used_lan = []
        for row in df[:max_rows].values:
            used_lan.append(detect_langs(row[0]))
        return used_lan

    def lr_score(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df, df.label, test_size=0.3, random_state=42)
        filename = 'assets/models/logistic_regression_model.sav'

        count_vect = CountVectorizer(stop_words=self.stopwords)
        temp = count_vect.fit_transform(X_train['review'])

        tdif = TfidfTransformer()
        temp2 = tdif.fit_transform(temp)

        # load the model from disk
        model = pickle.load(open(filename, 'rb'))

        # save model
        # model = LogisticRegression(random_state=42)
        # model = model.fit(temp2, y_train)
        # pickle.dump(model, open(filename, 'wb'))

        prediction_data = tdif.transform(count_vect.transform(X_test['review']))
        predicted = model.predict(prediction_data)
        print('\nLogisticRegression confusion matrix:\n', confusion_matrix(y_test, predicted))
        print("LogisticRegression accuracy score:", accuracy_score(y_test, predicted))
        print("LogisticRegression recall score:", recall_score(y_test, predicted, pos_label=1))
        print("LogisticRegression precision score:", precision_score(y_test, predicted, pos_label=1))

        # Test if model works with fictional reviews
        print("This is a really beautiful product - prediction: ",
              model.predict(tdif.transform(count_vect.transform(["This is a really beautiful product"]))))
        print("This is a terrible product - prediction: ",
              model.predict(tdif.transform(count_vect.transform(["This is a terrible product"]))))

        # ROC Curve
        y_pred_proba = model.predict_proba(prediction_data)[::, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.title('Logistic Regression AUC')
        plt.show()

    def nb_score(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df, df.label, test_size=0.3, random_state=42)
        filename = 'assets/models/naive_bayes_model.sav'

        count_vect = CountVectorizer(stop_words=self.stopwords)
        temp = count_vect.fit_transform(X_train['review'])

        tdif = TfidfTransformer()
        temp2 = tdif.fit_transform(temp)

        # load the model from disk
        model = pickle.load(open(filename, 'rb'))

        # save model
        # model = MultinomialNB()
        # model.fit(temp2, y_train)
        # pickle.dump(model, open(filename, 'wb'))

        prediction_data = tdif.transform(count_vect.transform(X_test['review']))
        predicted = model.predict(prediction_data)
        print('\nNB confusion matrix:\n', confusion_matrix(y_test, predicted))
        print('NB score: ', accuracy_score(y_test, predicted))
        print("NB recall score:", recall_score(y_test, predicted, pos_label=1))
        print("NB precision score:", precision_score(y_test, predicted, pos_label=1))

        # Test if model works with fictional reviews
        print("This is a really beautiful product - prediction: ",
              model.predict(tdif.transform(count_vect.transform(["This is a really beautiful product"]))))
        print("This is a terrible product - prediction: ",
              model.predict(tdif.transform(count_vect.transform(["This is a terrible product"]))))

        # ROC Curve
        y_pred_proba = model.predict_proba(prediction_data)[::, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.title('Naïve Bayes AUC')
        plt.show()

    def rf_score(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df, df.label, test_size=0.3, random_state=42)
        filename = 'assets/models/random_forest_model.sav'

        count_vect = CountVectorizer(stop_words=self.stopwords)
        temp = count_vect.fit_transform(X_train['review'])

        tdif = TfidfTransformer()
        temp2 = tdif.fit_transform(temp)

        # load the model from disk
        model = pickle.load(open(filename, 'rb'))

        # save model
        model = RandomForestClassifier(n_estimators=200, max_depth=600, random_state=42)
        # model.fit(temp2, y_train)
        # pickle.dump(model, open(filename, 'wb'))

        # Search for best parameters to use on RandomForest model
        # param_grid = {
        #     'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 750],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth': [100, 150, 200, 250, 300, 350, 400, 450, 500],
        #     'criterion': ['gini', 'entropy']
        # }
        #
        # gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        # gs.fit(temp2, y_train)
        # print(gs.best_params_)

        prediction_data = tdif.transform(count_vect.transform(X_test['review']))
        predicted = model.predict(prediction_data)
        print('\nRandomForest confusion matrix:\n', confusion_matrix(y_test, predicted))
        print("RandomForest score:", accuracy_score(y_test, predicted))
        print("RandomForest recall score:", recall_score(y_test, predicted, pos_label=1))
        print("RandomForest precision score:", precision_score(y_test, predicted, pos_label=1))

        # Test if model works with fictional reviews
        print("This is a really beautiful product - prediction: ",
              model.predict(tdif.transform(count_vect.transform(["This is a really beautiful product"]))))
        print("This is a terrible product - prediction: ",
              model.predict(tdif.transform(count_vect.transform(["This is a terrible product"]))))

        # ROC Curve
        y_pred_proba = model.predict_proba(prediction_data)[::, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.title('Random Forest AUC')
        plt.show()
