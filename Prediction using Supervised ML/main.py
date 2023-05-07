from flask import Flask, render_template, request, redirect, url_for
import sklearn
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('Hours_Score_Model.pk1')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/success/<name>')
def success(name):
    return 'Score Prediction is %s' % name

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    hour = float(request.form.get('Hours'))
    score = model.predict(np.array(hour).reshape(-1, 1))
    print(score[0][0])
    rounded_score = round(score[0][0], 2)
    return redirect(url_for('success', name=str(rounded_score)))

if __name__ == '__main__':
    app.run()
