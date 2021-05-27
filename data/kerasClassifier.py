import pickle

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import time

import plotly.express as px

from numpy import asarray
from numpy import zeros
from tensorflow.python.keras.layers import Embedding, LSTM, Dropout, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from data.db_handler import DBHandler


class KerasClassifier:
    @staticmethod
    def text_classification(df):
        # File for storing history
        filename = '/kerasTrainHistory'

        # Preprocessing
        stop_words = set(stopwords.words('english'))
        stop_words.add(('Positive', 'Negative'))
        df['review'] = df['review'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

        X_train, X_test, y_train, y_test = train_test_split(df.review, df.label, test_size=0.3,
                                                            random_state=42)

        tokenizer = Tokenizer(num_words=5000, lower=True)
        tokenizer.fit_on_texts(X_train)
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        vocab_size = len(tokenizer.word_index) + 1

        MAXLEN = 100
        X_train = pad_sequences(X_train, padding='post', maxlen=MAXLEN)
        X_test = pad_sequences(X_test, padding='post', maxlen=MAXLEN)

        # Building model
        start = time.time()
        model = Sequential()

        model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=MAXLEN))

        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

        model.add(LSTM(150, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(MaxPooling1D(5))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(3))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Load model to save time
        model = load_model('kerasmodel')

        # Train model
        # history = model.fit(X_train, y_train, batch_size=128, epochs=3, verbose=1, validation_data=(X_test, y_test),
        #                     validation_split=0.2)
        history = pickle.load(open(filename, 'rb'))
        print(history)
        # pickle.dump(history.history, open(filename, 'wb'))
        end = time.time()

        build_time = end - start

        # Save model to save some time
        # model.save('kerasmodel')

        # Print out results
        print("\nAmount of seconds for Keras to finish model: %.2f" % build_time)
        loss, accuracy = model.evaluate(X_train, y_train, verbose=1)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print("Testing Accuracy: {:.4f}".format(accuracy))

        # Building graphs
        accuracy_fig = px.line(x=[1, 2, 3], y=[history['accuracy'], history['val_accuracy']],
                               labels={"variable": "type", "x": "epoch"}, title="Model accuracy")
        accuracy_fig.data[0].name = "accuracy"
        accuracy_fig.data[0].hovertemplate = "type=accuracy<br>epoch=%{x}<br>value=%{y}"
        accuracy_fig.data[1].name = "validation accuracy"
        accuracy_fig.data[1].hovertemplate = "type=validation accuracy<br>epoch=%{x}<br>value=%{y}"
        accuracy_fig.update_layout(
            xaxis_title="epoch",
            yaxis_title="Accuracy"
        )

        loss_fig = px.line(x=[1, 2, 3], y=[history['loss'], history['val_loss']],
                           labels={"variable": "type", "x": "epoch"}, title="Model loss")
        loss_fig.data[0].name = "loss"
        loss_fig.data[0].hovertemplate = "type=loss<br>epoch=%{x}<br>value=%{y}"
        loss_fig.data[1].name = "validation loss"
        loss_fig.data[1].hovertemplate = "type=validation loss<br>epoch=%{x}<br>value=%{y}"
        loss_fig.update_layout(
            xaxis_title="epoch",
            yaxis_title="Loss"
        )

        return accuracy_fig, loss_fig
