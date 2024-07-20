from flask import Flask, jsonify, request
from joblib import load

presence_classifier = load('../model/yn_class.joblib')
presence_vect = load('../model/yn_vector.joblib')
category_classifier = load('../model/cat_class.joblib')
category_vect = load('../model/cat_vector.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        output = []
        data = request.get_json().get('tokens')

        for token in data:
            result = presence_classifier.predict(presence_vect.transform([token]))
            if result == 'Dark':
                cat = category_classifier.predict(category_vect.transform([token]))
                output.append(cat[0])
            else:
                output.append(result[0])

        dark = [data[i] for i in range(len(output)) if output[i] == 'Dark']
        for d in dark:
            print(d)

        message = '{ \'result\': ' + str(output) + ' }'
        print(message)

        json = jsonify(message)

        return json

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
