
from flask import Flask,render_template,request,flash,redirect,url_for,send_from_directory
from werkzeug.utils import secure_filename
from predict import predictImage
import cv2
import os
from keras.models import load_model

UPLOAD_FOLDER = 'upload_folder'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}
UPLOADED_FILE_NAME = "uploadedFile.jpg"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if os.listdir(app.config['UPLOAD_FOLDER'])!= []:
    os.remove(os.path.join(UPLOAD_FOLDER,UPLOADED_FILE_NAME))
    

model = load_model("model/model.h5")


def allowed_file(filename):
    print(filename)
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template("mainPage.html")
        file = request.files['file']
       
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return  render_template("mainPage.html") 

        if file and allowed_file(file.filename):
            
            #filename = secure_filename(file.filename)
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], UPLOADED_FILE_NAME))
            selectedImage,result = predict()
            return render_template("mainPage.html",result=result,selectedImage=selectedImage) 
    
   
    

def predict():
    
    imgPath = os.path.join(UPLOAD_FOLDER, UPLOADED_FILE_NAME)

    selectedImage = cv2.resize(cv2.imread(imgPath),(224,224))
        
    result =  predictImage(selectedImage,model)
        
   
    return imgPath,result
 
  

@app.route('/')
def home():
    return render_template("mainPage.html") 
    
        
if __name__ == '__main__':
    app.run(debug=True)

