import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


import pandas as pd 
import numpy as np 
import nltk
import re

import gensim

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dropout, Dense, Embedding
from keras import Sequential
from newspaper import Article

import preprocess_kgptalkie as ps
app = QApplication(sys.argv)

def search(url):
    article = Article(url,language= 'en')
    return article

def preprocess_data(x_test,tokenizer):
    unknown_pub = []
    for idx, row in enumerate(x_test.text.values):
        try:
            record = row.split('-', maxsplit=1)
            assert(len(record[0]) < 120)
        except:
            unknown_pub.append(idx)
    publisher = []
    tmp_text = []
    for idx, row in enumerate(x_test.text.values):
        if idx in unknown_pub:
            tmp_text.append(row)
            publisher.append('unknown')
        else:
            tmp_text.append(row)
            publisher.append('unknown')
    x_test['publisher'] = publisher
    x_test.text = tmp_text
    x_test.text = x_test.title + " " + x_test.text
    x_test.text = x_test.text.apply(lambda x:str(x).lower())
    x_test.text.apply(lambda x:ps.remove_special_chars(x))
    x_test_preprocessed = [row.split() for row in x_test.text.tolist()]
    x_test_preprocessed = tokenizer.texts_to_sequences(x_test_preprocessed)
    x_test_preprocessed = pad_sequences(x_test_preprocessed, maxlen = 1000)
    return x_test_preprocessed

class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.clicksCount = 0
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("News Detector")
        self.resize(1000, 563)
        self.setStyleSheet("background: qlineargradient( x1:0 y1:0, x2:1 y2:1, stop:0 #6eaeee, stop:1 #19507e);")
        self.def_style = """
            QTextEdit{ background:#6eaeee;border-radius:20px;color:dark-blue}
            QLineEdit{ background:#6eaeee;border-radius:10px;color:dark-blue}
            QPushButton{ background:#6eaeee;border-radius:10px;}
            QPushButton:hover{ background:#19507e}
        """
        self.url_input = QLineEdit(self)
        self.url_input.move(200,40)
        self.url_input.resize(430,50)
        self.url_input.setStyleSheet(self.def_style)
        self.url_input.setAlignment(Qt.AlignCenter)

        self.search_button = QPushButton("Search",self)
        self.search_button.move(650,40)
        self.search_button.resize(150,50)
        self.search_button.setStyleSheet(self.def_style)
        self.search_button.setCursor(QCursor(Qt.PointingHandCursor))

        self.inference = QTextEdit(self)
        self.inference.move(60,130)
        self.inference.resize(420,400)
        self.inference.setEnabled(False)
        self.inference.setAlignment(Qt.AlignCenter)
        self.inference.setStyleSheet(self.def_style)
        self.inference.setFont((QFont('Monospace',15)))

        self.output = QTextEdit(self)
        self.output.move(540,130)
        self.output.resize(420,50)
        self.output.setEnabled(False)
        self.output.setStyleSheet(self.def_style)
        self.output.setFont((QFont('Monospace',15)))
        self.output.setText('Output: ')
        self.output.setAlignment(Qt.AlignCenter)

        self.name = QTextEdit(self)
        self.name.move(540,220)
        self.name.resize(420,50)
        self.name.setEnabled(False)
        self.name.setStyleSheet(self.def_style)
        self.name.setFont((QFont('Monospace',15)))
        self.name.setText('Authors: ')
        self.name.setAlignment(Qt.AlignCenter)

        self.date = QTextEdit(self)
        self.date.move(540,310)
        self.date.resize(420,50)
        self.date.setEnabled(False)
        self.date.setStyleSheet(self.def_style)
        self.date.setFont((QFont('Monospace',15)))
        self.date.setText('Date: ')
        self.date.setAlignment(Qt.AlignCenter)

        self.search_button.clicked.connect(self.search)
        self.model = tf.keras.models.load_model('data/saved_model')

        hi = pd.read_csv('data/preprocessed_data.csv')
        x = [row.split() for row in hi.text.tolist()]
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(x)
    
    def search(self):
        self.url = self.url_input.text()
        try:
            article = search(self.url)
            article.download()
            article.parse()
            article.nlp()
            summary = article.summary
            name = article.authors
            date = article.publish_date
            self.inference.setText("Summary:\n"+summary)
            self.output.setStyleSheet('background:#e67f87;border-radius:10px;color:dark-blue')
            try:
                val = 'Author: '+name[0]
            except:
                val = 'Author: '
            self.name.setText(val)
            try:
                self.date.setText('Date: '+str(date))
            except:
                self.date.setText('Date: ')
            dict1 = {'title':[article.title],'text':[article.text]}
            df = pd.DataFrame(dict1)
            out = (self.model.predict(preprocess_data(df,self.tokenizer))>0.5)
            out = np.squeeze(out)
            if out:
                self.output.setStyleSheet('background:#66ff33;border-radius:10px;color:dark-blue')
            else:
                self.output.setStyleSheet('background:#e67f87;border-radius:10px;color:dark-blue')
            self.output.setText("Output: "+str(out))
        except Exception as err:
            self.inference.setText("Link not found")
            self.output.setText('Output: ')
            self.output.setStyleSheet(self.def_style)
            self.name.setText('Authors: ')
            self.date.setText('Date: ')
            print(err)

win = Window()
win.show()
sys.exit(app.exec())
