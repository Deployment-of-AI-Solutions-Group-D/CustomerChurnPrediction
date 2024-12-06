# pip install Flask

from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction import DictVectorizer

app = Flask(__name__)

with open('./model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('./dv.pkl', 'rb') as f:
    dv = pickle.load(f)

@app.route('/')
def hello_world():
    return 'Hello, Docker!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    transformed_data = dv.transform([data['input']])
    prediction = model.predict_proba(transformed_data)[0, 1]
    return jsonify({'churn_probability': prediction})


@app.route('/predict_get', methods=['GET'])
def predict_get():
    customer = {
        'gender': request.args.get('gender'),
        'seniorcitizen': int(request.args.get('seniorcitizen', 0)),
        'partner': request.args.get('partner'),
        'dependents': request.args.get('dependents'),
        'phoneservice': request.args.get('phoneservice'),
        'multiplelines': request.args.get('multiplelines'),
        'internetservice': request.args.get('internetservice'),
        'onlinesecurity': request.args.get('onlinesecurity'),
        'onlinebackup': request.args.get('onlinebackup'),
        'deviceprotection': request.args.get('deviceprotection'),
        'techsupport': request.args.get('techsupport'),
        'streamingtv': request.args.get('streamingtv'),
        'streamingmovies': request.args.get('streamingmovies'),
        'contract': request.args.get('contract'),
        'paperlessbilling': request.args.get('paperlessbilling'),
        'paymentmethod': request.args.get('paymentmethod'),
        'tenure': int(request.args.get('tenure', 0)),
        'monthlycharges': float(request.args.get('monthlycharges', 0.0)),
        'totalcharges': float(request.args.get('totalcharges', 0.0))
    }
    transformed_data = dv.transform([customer])
    prediction = model.predict_proba(transformed_data)[0, 1]
    return jsonify({'churn_probability': prediction})

@app.route('/ui')
def ui():
    return app.send_static_file('dashboard.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8085, debug=False)
