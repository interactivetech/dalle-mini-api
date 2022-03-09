import os
# Using Flask since Python doesn't have built-in session management
from flask import Flask, session, render_template, request, jsonify
# Our target library
import requests
import json
from pathlib import Path
from base64 import encodebytes
from PIL import Image
import io
import numpy as np
from model import load_model, generate_images, load_model2

import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
from time import strftime
import traceback

model,tokenizer,vqgan,clip,processor,model_params, vqgan_params, clip_params = load_model2()
app = Flask(__name__)

# def get_response_image(image_path):
#     pil_img = Image.open(image_path, mode='r') # reads the PIL image
#     byte_arr = io.BytesIO()
#     pil_img.save(byte_arr, format='jpeg') # convert the PIL image to byte array
#     encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
#     return encoded_img
def get_response_image(im):
    byte_arr = io.BytesIO()
    im.save(byte_arr, format='jpeg') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route('/')
def index():
    return "Welcome to dalleapi.com"

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == "POST":
        try:
            input_json = request.get_json(force=True) 
            print("Prompt: {}".format(input_json['prompt']))
            timestamp = strftime('[%Y-%b-%d %H:%M]')
            app.logger.info('%s %s %s %s %s %s',
                            timestamp,
                            request.remote_addr,
                            request.method,
                            request.scheme,
                            request.full_path,
                            input_json['prompt'])
            images =generate_images(
                input_json['prompt'],
                input_json['n_images'],
                input_json['gen_top_k'],
                model,
                tokenizer,
                vqgan,
                clip,
                processor,
                model_params, 
                vqgan_params, 
                clip_params)
            encoded_images = [get_response_image(i) for i in images]
            # encoded_imges = []
            # for image_path in range(10):
            #     encoded_imges.append(get_response_image_test())
            return jsonify({'prompt':input_json['prompt'],
                            'n_images':input_json['n_images'],
                            'gen_top_k':input_json['gen_top_k'],
                            'result': encoded_images})
        except:
            return jsonify({"sorry": "Sorry, no results! Please try again."}), 500

@app.after_request
def after_request(response):
    timestamp = strftime('[%Y-%b-%d %H:%M]')
    app.logger.info('%s %s %s %s %s %s', timestamp, request.remote_addr, request.method, request.scheme, request.full_path, response.status)
    return response

# @app.errorhandler(Exception)
# def exceptions(e):
#     tb = traceback.format_exc()
#     timestamp = strftime('[%Y-%b-%d %H:%M]')
#     app.logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s', timestamp, request.remote_addr, request.method, request.scheme, request.full_path, tb)
#     return e.status_code

if __name__ == '__main__':
    handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=3)
    # logger = logging.getLogger('tdm')
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run(
        host="0.0.0.0",
        port=80
    )