# Lung server
import flask

from src.routes import *
app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hom():
    return home()


@app.route('/info', methods=['GET'])
def inf():
    return info()


if __name__ == '__main__':
    app.run(port=5002)
