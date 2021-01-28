from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename1 = r'C:\Users\admin\Desktop\cv-transform.pkl'
filename2 = r'C:\Users\admin\Desktop\spam-sms-mnb-model.pkl'
classifier = pickle.load(open(filename2, 'rb'))
cv = pickle.load(open(filename1, 'rb'))
app = Flask(__name__)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)