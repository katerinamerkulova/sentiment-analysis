# Загружаем модель и делаем предсказание
__author__ = 'xead'
from codecs import open
import time
from flask import Flask, render_template, request
import sentiment_analysis as sa
import re

app = Flask(__name__)
@app.route("/", methods=["POST", "GET"])
def index_page(text="", prediction_message=""):
    if request.method == "POST":
        text = request.form["text"]
        text = re.sub(r'[^\w]', ' ', text).lower()
        logfile = open("ydf_demo_logs.txt", "a", "utf-8")
        pred = sa.predict([text])
        if pred == ['neg']:
            prediction_message = 'negative'
        else:
            prediction_message = 'positive'
        print(text)
        print("<response>", file=logfile)
        print(text, file=logfile)
        print(prediction_message)
        print(prediction_message, file=logfile)
        print("</response>", file=logfile)
        logfile.close()
    return render_template('sentiment.html', text=text, prediction_message=prediction_message)


if __name__ == "__main__":
    app.run()