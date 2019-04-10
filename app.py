from flask import Flask,render_template, request,jsonify,Response, redirect,url_for
import requests
import json
import sys
import time
import numpy as np
from filestore.k8.ner_service.ner_predict import prepare_model
from filestore.k8.ner_service.ner_predict import make_prediction
from filestore.k8.ner_service.flask_predict import get_prediction
app = Flask(__name__)
# print(a)
print('Please wait while the Fine-Tuned BERT model is loaded. ')
processor, label_list, tokenizer, estimator = prepare_model()
print('*** Model is successfully loaded ***')

@app.route("/bert_ner", methods = ['POST', 'GET'])
def details():
    if request.method == 'POST':
        user_text = request.json['text']
    output = get_prediction(user_text,processor, label_list, tokenizer, estimator)

    # for key, i in enumerate(l):
    #     output[str(key)] = str(i[0])
    print(output)
    new_output=get_seq_attributes(output)
    return jsonify({'response':new_output}), 201

##Function to get sequnce attributes giving a NER dictionary
def get_correct_attribute(target,data):
    temp_seq = ''
    temp_label = data[target]
    temp_index = -1
    for i in list(data.keys())[list(data.keys()).index(target):]:
        try:
            if data[i] == temp_label:
                temp_seq = temp_seq+' '+i

            else:
                temp_index = list(data.keys()).index(i)
                break
        except:
            pass
    return [[temp_seq.strip(),temp_label],temp_index]
def get_seq_attributes(data):
    output={}
    target = list(data.keys())[0]
    flag = 0
    while flag!=1:
        seq,next_index = get_correct_attribute(target,data)
        target = list(data.keys())[next_index]

        try:
            output[seq[1]].append(seq[0])
        except:
            output[seq[1]] = [seq[0]]

        if next_index == -1:
            flag = 1
    try:
    	del output['X']
    except:
        pass
    try:

        del output['O']
    except:
        pass
    return output




if __name__ == "__main__":
    app.run(debug = False,host='0.0.0.0',port=5002)
