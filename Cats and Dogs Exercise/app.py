
from flask import Flask,render_template,request,flash,redirect
from werkzeug.utils import secure_filename
from predict import predictImage
import cv2
from keras.models import load_model



app = Flask(__name__)



model = load_model("model/model.h5")


def  upload():
    imagePath=request.form.get('image')

    print("****************")
    print(imagePath)
    print("****************")

    selectedImage = cv2.resize(cv2.imread(imagePath),(224,224))
    return selectedImage
    
@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        selectedImage=upload()
        
        result =  predictImage(selectedImage,model)
        
        print("Predict kullanarak gönderdim "+result)
        
    
    return render_template("mainPage.html",result=result) 

@app.route('/',methods=["GET"])
def home():
    print("Home kullanarak gönderdim ")
    return render_template('mainPage.html',) 
    
        
if __name__ == '__main__':
    app.run(debug=True)

