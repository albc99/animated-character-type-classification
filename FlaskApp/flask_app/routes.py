from flask import current_app as app
from flask import render_template, redirect, request, session, url_for, copy_current_request_context, abort, Flask, flash
from flask_socketio import SocketIO, emit, join_room, leave_room, close_room, rooms, disconnect
#from .utils.database.database  import database
from werkzeug.datastructures   import ImmutableMultiDict
from werkzeug.utils import secure_filename
from pprint import pprint
import json
import random
import functools
from . import socketio
import pandas as pd
import os, shutil
import pickle
from torchvision import datasets, transforms
import torchvision.models as models
import torch
import io

@app.route('/')
def root():
    return redirect('/pokemontypeprediction')


@app.route('/pokemontypeprediction')
def pokemontypeprediction():
    return render_template('pokemontypeprediction.html')


@app.route('/pokemonclassificationresult', methods = ['GET', 'POST'])
def pokemonclassificationresult():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    # check if the post request has the file part
    if 'image' not in request.files:
        flash('No file part')
        return render_template('pokemontypeprediction.html')
    file = request.files['image']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected image')
        return render_template('pokemontypeprediction.html')
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        this_type, this_color = make_prediction()
        return render_template('pokemonclassificationresult.html', prediction=this_type, type_color=this_color)

@app.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def make_prediction():
    with open('flask_app/static/models/resnet_pokemon_image_classifier_model.pkl', 'rb') as file:
        class_names = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass',
                       'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']
        colors = ['#94BC4A', '#736C75', '#6A7BAF', '#E5C531', '#E397D1', '#CB5F48', '#EA7A3C', '#7DA6DE', '#846AB6',
                  '#71C558', '#CC9F4F', '#70CBD4', '#AAB09F', '#B468B7', '#E5709B', '#B2A061', '#89A1B0', '#539AE2']
        resnet152 = CPU_Unpickler(file).load()
        data_dir = app.config['UPLOAD_FOLDER'][:app.config['UPLOAD_FOLDER'].rfind('/')]

        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        test_data = datasets.ImageFolder(data_dir, transform=transform)
        X, y = test_data[0]
        y_pred = resnet152(X.to("cpu")[None, ...])
        y_pred = y_pred.argmax(1)
        return class_names[y_pred], colors[y_pred]

