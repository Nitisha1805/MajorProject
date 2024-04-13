from flask import Flask, render_template,redirect,request, flash, session
from database import add_to_db, open_db
from werkzeug.utils import secure_filename
from common.files_utils import *
import pandas as pd
from joblib import load

app = Flask(__name__)
app.secret_key = 'thisissupersecretkeyfornoone'

# load model
def load_model():
    try:
        model = load(r'static\models\GaussianNB.jb')
        return model
    except Exception as e:
        print(e)
        return None
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        keys = request.form.keys()
        values = request.form.values()
        # create a dataframe
        df = pd.DataFrame([values], columns=keys)
        print(df)
        # load model
        model = load_model()
        if model:
            prediction = model.predict(df)
            print(prediction)
            flash('Prediction completed', 'success')
            return render_template('result.html', prediction=prediction)
        else:
            flash('Model not loaded', 'danger')
            return redirect('/prediction')
    df = pd.read_csv('data/train.csv', nrows=1)
    columns = tuple(df.iloc[0].items())

    return render_template('prediction.html', columns = columns)


@app.route('/dataset/view')
def view_dataset():
    file = 'data/train.csv'
    df = pd.read_csv(file, nrows=100)
    return render_template('view_dataset.html', data=df.to_html(classes='table table-striped table-hover table-bordered'))




if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
 
 