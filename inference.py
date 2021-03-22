#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import json
import pickle as pkl


def load_files():
    with open('weights/department.json', 'r', encoding='utf-8') as data_file:    
        department= eval(json.load(data_file))

    with open('weights/region.json', 'r', encoding='utf-8') as region_file:    
        region= json.load(region_file)

    education= {"Below Secondary":0,"Bachelor's":1,"Master's & above":2}

    ss= pkl.load(open('weights/standardScaler.pkl','rb'))
    model = pkl.load(open('vot2.pkl', 'rb'))

    return department, region, education, ss, model
    

def predict_promoted(input_data, department, region, education, ss, model):

    # input_data= pd.DataFrame.from_dict(input_data,orient='index')

    # print(input_data[])
    # print(input_data.loc[[0]])
    input_data['department']= input_data['department'].map(department)
    input_data['education']= input_data['education'].map(education)
    input_data['region']= input_data['region'].map(region)


    if input_data.iloc[0]['gender']=='m':
        input_data['gender_m']= 1
    else:
        input_data['gender_m']= 0

    if input_data.iloc[0]['recruitment_channel']=='referred':
        input_data['recruitment_channel_referred']= 1
        input_data['recruitment_channel_sourcing']= 0
    elif input_data.iloc[0]['recruitment_channel']=='sourcing':
        input_data['recruitment_channel_referred']= 0
        input_data['recruitment_channel_sourcing']= 1
    else:
        input_data['recruitment_channel_referred']= 0
        input_data['recruitment_channel_sourcing']= 0


    input_data1= input_data.drop(columns=['gender','recruitment_channel',
                                          'employee_id'])

    num_sca=ss.transform(input_data1)
    X = pd.DataFrame(num_sca, columns = input_data1.columns)

    ypred= model.predict(X)

    if ypred==0:
        return 'Not promoted' 
    else:
        return 'Promoted'


# df_test=pd.read_csv('Data/train.csv')
# df_test=df_test.set_index('employee_id')
# input_data= df_test.loc[[65438]]

# department, region, education, ss, model= load_files()
# print(predict(input_data, department, region, education, ss, model))
