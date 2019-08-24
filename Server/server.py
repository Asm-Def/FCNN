from flask import Flask, render_template, request, Response
import flask
import numpy as np
import json
import glob
import torch
import torch.nn as nn
import nibabel as nib
import math
from Models.FCNN import FCNN
import os
import cv2 as cv
import base64

app = Flask(__name__)
model = FCNN()
log_dir = os.path.join('log', 'FCNN')
model.load(log_dir)
if torch.cuda.is_available():
    model = model.cuda()

my_pass = '29jfqw0q4'

files = sorted(glob.glob('../Data/files/img/img*.nii.gz'))
images = [np.tanh(nib.load(file).get_data().transpose((2, 1, 0)) / 170.0) for file in files]
cur_file = None
image = None
tensor = None
predict = None


files = [(i, os.path.split(file)[-1]) for i, file in enumerate(files)]

# Gauss Filer
tmp = torch.empty((5, 5), dtype=torch.float32)
for i in range(5):
    for j in range(5):
        t = (i-2)**2 + (j-2)**2
        tmp[i,j] = math.exp(- t / 8) / 8 / math.pi
gauss_filter = nn.Conv2d(1, 1, 5, padding=2, bias=False)
gauss_filter.weight = nn.Parameter(tmp.view(1, 1, 5, 5), False)
if torch.cuda.is_available():
    gauss_filter = gauss_filter.cuda()

del tmp


@app.route('/', methods=['GET'])
def index():
    password = request.args.get('token')
    if password == my_pass:
        return render_template('index.html', args=(files, password))
    else:
        flask.abort(404)


def wrap_tensor(fore_clicks, back_clicks):
    global tensor, image
    assert not (image is None)

    img = torch.from_numpy(image).to(dtype=torch.float32)  # (x, y)
    zeros = torch.zeros_like(img)

    fore = zeros
    for click in fore_clicks:
        fore[click[0], click[1]] = 1.0

    back = zeros
    for click in back_clicks:
        back[click[0], click[1]] = 1.0

    if torch.cuda.is_available():
        img = img.cuda()
        fore = fore.cuda()
        back = back.cuda()

    img = img.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        fore = gauss_filter(fore.unsqueeze(0).unsqueeze(0))
        back = gauss_filter(back.unsqueeze(0).unsqueeze(0))

    clicks = torch.cat((fore, back), 1)
    clicks /= clicks.max()

    tensor = torch.cat((img, clicks), 1)


def wrap_image():  # 将预测结果拼接成用来显示的图像
    global image, predict
    assert not image is None
    img = ((image + 1) * (255 / 2)).astype(np.uint8)
    area = (predict * 255).astype(np.uint8)

    img = np.stack((img, img, img), 2)
    zeros = np.zeros_like(area)
    area = np.stack((area, zeros, zeros), 2)
    print(img.dtype, area.dtype)
    img = cv.cvtColor(cv.addWeighted(img, 0.7, area, 0.3, 0.0), cv.COLOR_RGB2BGR)

    print(img.shape)

    cv.imwrite('tmp.png', img)


@app.route('/img', methods=['GET'])
def get_image():
    global images, image, tensor, cur_file, predict
    password = request.args.get('token')
    if password == my_pass:
        cur_file = int(request.args.get('file'))
        image = images[cur_file][0]
        predict = np.zeros(image.shape)
        wrap_image()

        return render_template('image.html', channels=images[cur_file].shape[0], token=password)
    else:
        flask.abort(404)


@app.route('/api/modify-channel/<int:ch_id>', methods=['POST'])
def mod_channel(ch_id):
    global images, image, tensor, cur_file, predict
    password = request.form['token']
    print('password=', password)
    print("ch_id=", ch_id)
    print("len=", images[cur_file].shape[0])
    if password == my_pass and 0 <= ch_id < int(images[cur_file].shape[0]):
        image = images[cur_file][ch_id]
        return "OK"

    else:
        flask.abort(404)


@app.route('/api/get-predict', methods=['POST', 'GET'])
def get_pred():
    global tensor, predict, image
    if request.method == "POST":
        password = request.form.get('token')
        fore = json.loads(request.form.get('fore_click'))  # [[xi, yi] for i]
        back = json.loads(request.form.get('back_click'))

    else:
        password = request.args.get('token')
        fore = json.loads(request.args.get('fore_click'))
        back = json.loads(request.args.get('back_click'))

    if password != my_pass:
        flask.abort(404)
        return

    if len(fore) + len(back) == 0:
        predict = np.zeros_like(image)
    else:
        wrap_tensor(fore, back)
        with torch.no_grad():
            predict = model(tensor).cpu().numpy().reshape(image.shape)

    wrap_image()

    img_file = open('tmp.png', 'rb')
    resp = Response(img_file, mimetype='image/png')
    print(resp)
    return resp


app.run(host='0.0.0.0', port=80)
