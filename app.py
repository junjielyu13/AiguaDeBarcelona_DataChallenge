from flask import Flask, render_template
from flask_cors import *

app = Flask(__name__, template_folder="./templates", static_folder='./static')
CORS(app, resources=r'/*')


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="localhost", port="8080", debug=True)