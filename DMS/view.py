from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
def home(request):
    return render (request, 'home.html')
def predict(request):
    return render (request, 'predict.html')
def result(request):
    data = pd.read_csv(r'C:\diabetes.csv')
    

    X = data.drop(columns='Outcome', axis=1)
    Y = data['Outcome']
    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=1)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train, y_train)

    vali1=float(request.GET['n1'])
    vali2=float(request.GET['n2'])
    vali3=float(request.GET['n3'])
    vali4=float(request.GET['n4'])
    vali5=float(request.GET['n5'])
    vali6=float(request.GET['n6'])
    vali7=float(request.GET['n7'])
    vali8=float(request.GET['n8'])
    input_data = np.asarray(([vali1,vali2,vali3,vali4,vali5,vali6,vali7,vali8]))

    # reshaping the array ( as we are predicting for one instance)
    input_data = input_data.reshape(1, -1)
    # standardize the input data
    std_data = scalar.transform(input_data)

    pred = classifier.predict(std_data)

    result1=""
    if pred==[1]:
        result1="Possitive"
    else:
        result1="Negative"

    return render (request, 'predict.html', {"result2": result1})