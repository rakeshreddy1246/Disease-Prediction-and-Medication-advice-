from flask import Flask, request, render_template,redirect
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model=pickle.load(open('model.pkl', 'rb'))
model1=pickle.load(open('model1.pkl','rb'))
dis=['Acne', 'Allergy', 'Diabetes', 'Fungal infection',
       'Urinary tract infection', 'Malaria', 'Migraine', 'Hepatitis B',
       'AIDS']

test=pd.read_csv("test_data.csv")
x_test=test.drop('prognosis',axis=1)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/disease')
def disease():
    return render_template('index.html')

@app.route('/drug1')
def drug1():
    return render_template('result.html')

@app.route('/drugresult',methods=['POST','GET'])
def drugresult():
    if request.method=='POST':
        l=[]
        Disease=request.form.get('Disease')
        for i in range(len(dis)):
            if dis[i]==Disease:
                l.append(float(i))
        Gender=request.form.get('Gender')
        if(Gender=='Male' or 'male'):
            l.append(float(1))
        else:
           l.append(float(0)) 
        age=request.form.get('age')
        l.append(float(age))
        l=np.array(l)
        l=np.array(l).reshape(1,-1)
        pr=model1.predict(l)
        return render_template('result.html',pr="The drug {}".format(pr))

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        col=x_test.columns
        inputt = [str(x) for x in request.form.values()]

        b=[0]*132
        for x in range(0,132):
            for y in inputt:
                if(col[x]==y):
                    b[x]=1
        b=np.array(b)
        b=b.reshape(1,132)
        prediction = model.predict(b)
        prediction=prediction[0]
    return render_template('index.html', pred="The probable diagnosis says it could be {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
