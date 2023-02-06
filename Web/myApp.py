import os
import json
from statistics import mode
import numpy as np
import pickle
from flask import Flask, render_template, redirect, url_for, request, session, jsonify
from textwrap import indent
app = Flask(__name__)
import pandas as pd
import os

app.secret_key = 'AshbornIsLegend'
listofdata = []

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    to_predict = pd.DataFrame(to_predict)
    print(to_predict)

    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))
    
    with open(f'{cwd}/models/DTCmodel.pkl' , 'rb') as f: #'C:/Users/Jegadit/Desktop/root/college/sem5/CSEB/11Assignments/19cse305/Project/ensemble/models/DTCmodel.pkl'
        DTC = pickle.load(f)
    with open(f'{cwd}/models/RFCmodel.pkl' , 'rb') as f:
        RFC = pickle.load(f)
    with open(f'{cwd}/models/KNNmodel.pkl' , 'rb') as f:
        KNN = pickle.load(f)
    with open(f'{cwd}/models/LRmodel.pkl' , 'rb') as f:
        LR = pickle.load(f)
    with open(f'{cwd}/models/GNBmodel.pkl' , 'rb') as f:
        GNB = pickle.load(f)
    with open(f'{cwd}/models/SVMmodel.pkl' , 'rb') as f:
        SVM = pickle.load(f)
    
    fina = [LR.predict(to_predict), 
            
            DTC.predict(to_predict), 
            
            RFC.predict(to_predict),

            KNN.predict(to_predict), 
            
            GNB.predict(to_predict), 
            
            SVM.predict(to_predict)]
    print(fina)
    return fina
    # loaded_model = pickle.load(open('RFCmodel.pkl','rb'))
    # result = loaded_model.predict(to_predict)
    # return result

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == "POST":

        listofdata.append(int(request.form["q1"]))
        listofdata.append(int(request.form["q2"]))
        listofdata.append(int(request.form["q3"]))
        listofdata.append(int(request.form["q4"]))
        listofdata.append(int(request.form["q5"]))
        listofdata.append(int(request.form["q6"]))
        listofdata.append(int(request.form["q7"]))
        listofdata.append(int(request.form["q8"]))
        listofdata.append(int(request.form["q9"]))
        listofdata.append(int(request.form["q10"]))
        listofdata.append(int(request.form["q11"]))
        listofdata.append(int(request.form["q12"]))
        listofdata.append(int(request.form["q13"]))
        listofdata.append(int(request.form["q14"]))
        listofdata.append(int(request.form["q15"]))
        listofdata.append(int(int(request.form["q16"])//int(request.form["q17"])))#//int(request.form["q17"]))

        print(listofdata)

        return redirect(url_for("user"))
    else:
        return render_template("index.html")


        
@app.route("/user")
def user():
    data = [
            ("LogisticRegression",100*0.7990338164251207 ),
            ("DecisionTreeClassifier",100*0.7753623188405797 ),
            ("RandomForestClassifier",100*0.8043478260869565 ),
            ("KNeighborsClassifier",100*0.7893719806763285 ),
            ("GaussianNB",100*0.7951690821256039 ),
            ("SVM",100*0.7763285024154589 )
        ]

    labels = [row[0] for row in data]
    values = [row[1] for row in data]

    print(labels)
    print(values)
    verd = ValuePredictor(listofdata)
    print(verd)
    max = mode([verd[0][0],verd[1][0],verd[2][0],verd[3][0],verd[4][0],verd[5][0]])
    return render_template("graph.html", labels = labels, values = values, max = max,  pred = verd, da = listofdata)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
