from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'UPLOAD_FOLDER'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('beluga_model')
model = model.to(device)

transformer = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return torch.Tensor(image)


def predict(PATH):
    pic = image_loader(transformer,PATH)
    pic = pic.to(device)
    pred = model(pic).to('cpu')
    k = np.argmax(pred.detach().numpy())
    print(k)
    return k

@app.route("/")
def home():
    return redirect(url_for('upload_file'))

@app.route("/pred/<filename>")
def pred(filename):
    pred = predict(f'UPLOAD_FOLDER/{filename}')
    k = "IDK"
    if pred == 1:
        k = 'You are not Begula'
    if pred == 0:
        k = 'Congratulations... You indeed are Begula'
    os.remove(f"UPLOAD_FOLDER/{filename}")
    return k

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('pred',filename=filename))
    return '''
    <!doctype html>
    <title>Upload your Picture</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''



if __name__ == "__main__":
    app.run()