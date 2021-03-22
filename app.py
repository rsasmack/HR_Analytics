import numpy as np
from flask import Flask, request, jsonify, render_template
from inference import load_files, predict_promoted
import pandas as pd

app = Flask(__name__) #Initialize the flask App

department, region, education, ss, model= load_files()
input_labels= ['department', 'region', 'education', 'gender',
               'recruitment_channel', 'no_of_trainings',
               'age' ,'previous_year_rating', 'length_of_service',
               'awards_won', 'avg_training_score']

@app.route('/')  # homepage
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    
    #For rendering results on HTML GUI
    
    if request.method == "POST":
        form_data= list(request.form.values())

        d= request.form.to_dict()
        print(d)

        df = pd.DataFrame([d.values()], columns=d.keys())
        # print(form_data[1:])
        # input_data= {}

        # for i in range(1, len(form_data)):
        #     input_data[input_labels[i-1]]= form_data[i]

        prediction= predict_promoted(df, department, region, education, ss,
                                     model)

    return render_template('index.html',
                           prediction_text='Predicted Class: {}'
                           .format(prediction))
# rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)
