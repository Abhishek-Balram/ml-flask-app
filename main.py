import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

from model_files.ml_model import predict_mpg


app = Flask('mpg_prediction')
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    vehicle = request.get_json()
    print(vehicle)
    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = predict_mpg(vehicle, model)

    result = {
        'mpg_prediction': list(predictions)
    }
    return jsonify(result)

@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model!!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)