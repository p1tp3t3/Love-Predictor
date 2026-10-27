from flask import Flask, jsonify, request, render_template
from model.love_predictor import LovePredictor
from model.recommender import Recommender, RecommenderUnavailable

app = Flask(__name__)
love_predictor = LovePredictor()
recommender = Recommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/api/model/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        prediction = love_predictor.predict(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify(prediction)

@app.route('/api/model/recommend', methods=['POST'])
def recommend():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        improvements = love_predictor.improvements(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    try:
        return jsonify(recommender.recommend(improvements, data['age']))
    except RecommenderUnavailable as e:
        app.logger.error('Could not generate tips: %s', e)
        return jsonify({'error': 'Tips are not available right now'}), 503

@app.route('/api/model/evaluate', methods=['GET'])
def evaluate():
    return jsonify({'accuracy': love_predictor.evaluate()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
