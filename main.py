import pickle

from waitress import serve
import numpy as np
import pandas as pd
from sklearn. ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from flask import Flask,render_template,request
from sklearn. compose import ColumnTransformer
from sklearn. preprocessing import LabelEncoder, OrdinalEncoder

main = Flask(__name__,template_folder='template')

@main.route("/")
def home():

  
    data = pd.read_csv('survey.csv')

    data.drop(['comments','state','Country','Timestamp'],axis=1, inplace=True)
    data['self_employed'].fillna('NO',inplace=True)
    data['work_interfere'].fillna('N/A',inplace=True)
    data.drop(data[(data['Age']>60) | (data['Age']<18)].index, inplace=True)
    data['Gender'].replace(['Cis Male','cis male','Cis Man','M','m','Man','mal',
                        'Mail','maile','Make','Mal','Male','male','Male(CIS)','Male (CIS)','Male-ish','msle','Malr'],'Male',inplace=True)
    data['Gender'].replace(['Female','female','F','f','Woman','cis-female/femme','Femake','Female (cis)','woman','femail','Female'],'Female',inplace=True)
    data['Gender'].replace(['Agender','All','Androgyne','Enby','Female (trans)',
                        'Female (trans)','fluid','Genderqueer',
                        'Guy (-ish) ^_^','male leaning androgynous','Nah','Neuter','non-binary',
                        'ostensibly male, unsure what that really means','p','queer','queer/she/they','something kinda male?',
                        'Trans woman','Trans-female'],'Other',inplace=True)
    x=data.drop('treatment', axis=1)
    y=data['treatment']
    data['treatment'].value_counts()

    ct=ColumnTransformer([('oe',OrdinalEncoder(),['Age', 'Gender', 'self_employed', 'family_history',
       'work_interfere', 'no_employees', 'remote_work', 'tech_company',
       'benefits', 'care_options', 'wellness_program', 'seek_help',
       'anonymity', 'leave', 'mental_health_consequence',
       'phys_health_consequence', 'coworkers', 'supervisor',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence'])],remainder='passthrough')
    x=ct.fit_transform(x)
    le=LabelEncoder()
    y=le.fit_transform(y)

    import joblib
    joblib.dump(ct,'feature_values')

    # split data
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=.3,random_state=49)
    model_dict = {}
    model_dict['Ada Boost Classifier']= AdaBoostClassifier(random_state=49)
    def model_test(x_train, x_test, y_train, y_test, model, model_name ):
        model.fit (x_train,y_train)
        y_pred=model.predict(x_test)
       
   
    abc =AdaBoostClassifier(random_state=99)
    abc.fit(x_train,y_train)
    pred_abc=abc.predict(x_test)

    from sklearn.model_selection import RandomizedSearchCV
    params_abc = {'n_estimators': [int(x) for x  in np.linspace(start = 1, stop =50, num = 15)],
             'learning_rate':[(0.97 + x / 100) for x in range(0, 8)]
             }
    abc_random = RandomizedSearchCV(random_state=49, estimator=abc, param_distributions = params_abc, n_iter=50, cv=5,n_jobs=-1)
    abc_tuned =AdaBoostClassifier(random_state=49,n_estimators=11, learning_rate=1.02)
    abc_tuned.fit(x_train,y_train)
    pred_abc_tuned=abc_tuned.predict(x_test)
    pickle.dump('abc_tuned',open('modell.pkl','wb'))




    return render_template("Mental_health_prediction.html")

@main.route("/predict",methods=['GET','POST'])
def predict():
    Age = request.form['Age']
    Gender = request.form['Gender']
    self_employed = request.form['self_employed']
    family_history = request.form['family_history']
    work_interfere = request.form['work_interfere']
    no_employees = request.form['no_employees']
    remote_work = request.form['remote_work']
    tech_company = request.form['tech_company']
    benefits = request.form['benefits']
    care_options = request.form['care_options']
    wellness_program = request.form['wellness_program']
    seek_help = request.form['seek_help']
    anonymity = request.form['anonymity']
    leave = request.form['leave']
    mental_health_consequence = request.form['mental_health_consequence']
    phys_health_consequence = request.form['phys_health_consequence']
    coworkers = request.form['coworkers']
    supervisor = request.form[' supervisor']

    mental_health_interview = request.form['mental_health_interview']
    phys_health_interview = request.form['phys_health_interview']
    mental_vs_physical = request.form[' mental_vs_physical']
    obs_consequence = request.form['obs_consequence']
    model = pickle.load(open("modell.pkl", "rb"))


    form_array=np.array([['Age','Gender','self_employed','family_history',
       'work_interfere','no_employees','remote_work','tech_company',
       'benefits','care_options','wellness_program','seek_help',
       'anonymity','leave','mental_health_consequence',
       'phys_health_consequence','coworkers','supervisor',
       'mental_health_interview','phys_health_interview',
       'mental_vs_physical','obs_consequence']])
    prediction = model.predict(form_array)[0]

    if prediction == 'Yes':
        result = "u have mental health problem"
    else:
        result = "your unfortunatly ok the issue is that the person with no mentalilness is the main  psycho"

    return render_template('result.html', result=result)
if __name__ == "__main__":
  serve(main, host="0.0.0.0", port=50200,threads=1)

