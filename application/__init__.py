from flask import Flask, request, Response, json
import numpy as numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

#load data
df = pd.read_csv("./carData/car.data", header = None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'eval'])
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#create onehotencoder and transform data
onehotencoder = OneHotEncoder()
ohc = onehotencoder.fit(X)
X = ohc.transform(X)

#train model
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, y)

#create flask instance
app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST'])
def predict(): 
    #get the data from request
    data = request.get_json(force=True)
    requestData = numpy.array([data["buying"], data["maint"], data["doors"], data["persons"], data["lug_boot"], data["safety"]])
    requestData = numpy.reshape(requestData, (1, -1))
    
    #get onehotencoding for input_data
    requestData = ohc.transform(requestData) 
    
    #Make prediction using model
    prediction = rfc.predict(requestData)
    return Response(json.dumps(prediction[0]))

if __name__ == '__main__':
    app.run()
