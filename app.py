import flask
import flask.views
from flask import request
from emotionRecognition import get_playlist
from emotionRecognition import get_emotion_grid
import numpy as np
from PIL import Image
import re
from io import BytesIO
import base64

app = flask.Flask(__name__)
app.secret_key = "Hello World!"

songs = list()


@app.route('/')
def landing():
    return flask.render_template("landing.html")


@app.route('/about')
def about():
    return flask.render_template("about.html")


@app.route('/team')
def team():
    return flask.render_template("ourTeam.html")


@app.route('/index')
def index():
    return flask.render_template("musi.html", songs=[])


@app.route('/hook', methods=['POST'])
def get_image():
    # convert base64 image and save it
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64)
    image_PIL = Image.open(BytesIO(base64.b64decode(image_data)))
    image_PIL.save("snapshots/myPicW.png", mode='RGB')
    songs = get_playlist()

    # printing the list of songs
    print("===================================================")
    print("Below is list of the songs:-")
    print()
    for x in range(len(songs)):
        print(x, ": ", songs[x], sep='')
    print()
    # print(songs)
    return flask.render_template("musi.html", songs=songs)


@app.route('/graph')
def get_graph():
    # draw emotion grid
    get_emotion_grid()
    # songs = get_playlist()
    return flask.render_template("musi.html", songs=songs)


if __name__ == '__main__':
    app.run(debug=True)
