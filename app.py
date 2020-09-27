from flask import Flask, request, jsonify,render_template
import pickle
app=Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/getSentiment',methods=['POST'])
def home():
    review = request.form['content']
    model1 = pickle.load(open('model.pkl', 'rb'))
    predictstr=model1.predict([review])
    print("Test Predict for ",review," is ", str(predictstr))
    print(type(str(predictstr)))
    if str(predictstr)=="['not happy']":
        return "NOT HAPPY"
    else:
        return "HAPPY"

if __name__ == '__main__':
    app.run(debug=True)