from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lol/<name>')
def lol(name):
    return name

if __name__ == '__main__':
    app.run()