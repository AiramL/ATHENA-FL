import os 
import time


from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pickle import load

UPLOAD_FOLDER = "./models/client_models"
MAX_CONTENT_LENGTH = 512 * 1000 * 1000
TEMPLATE_FOLDER = "template"
STATIC_FOLDER = "static"


app = Flask(__name__,template_folder=TEMPLATE_FOLDER,static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH



@app.route("/")
def index():
    #return render_template('athena_fl.html')
    return render_template('index.html')

@app.route("/about/")
def about():
    #return render_template('about.html')
    return render_template('index.html')

# Page to the client download the test model
@app.route('/download_model')
def download_test_model():
    return render_template('download_model.html')

# Page to return the client group 
@app.route('/client_group')
def client_group():
    #client_model_name = os.listdir(UPLOAD_FOLDER)[0]
    client_model_name = 'client-1-weights'
    with open(UPLOAD_FOLDER+'/'+client_model_name,"rb") as model:
        client_weights = load(model)[261]

    with open('./models/cluster_model/model','rb') as cluster:
        cluster_model = load(cluster)
    
    # classify the client model into a cluster
    return "Your group is "+str(cluster_model.predict([client_weights])).replace('[','').replace(']','')
    

@app.route('/upload', methods=['GET', 'POST'])
def upload_model():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('client_group', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
