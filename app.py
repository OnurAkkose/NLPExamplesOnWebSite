from flask import Flask,render_template,request,redirect,url_for
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/ozetle")    
def ozetle():
    return render_template("ozetle.html")
@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")
@app.route("/duygu", methods =["GET","POST"])
def analizEt():
    sent_model=open("duygu_analiz_model.pkl","rb")
    snt=joblib.load(sent_model)
    classes=np.array(["Olumsuz","Olumlu"])
    if request.method =="POST":
        text = request.form["sentiment_metin"]
        data =[text]
        sonuc = snt.predict(data).astype(int)
        return render_template("sentiment.html", prediction = classes[sonuc])
    else:

        return redirect(url_for("sentiment"))
    
@app.route("/tahmin",methods= ["GET","POST"])
def siniflandir():
    model=open("gazete_model.pkl","rb")
    clf=joblib.load(model)
    classes=np.array(["Magazin","Dünya","Spor","Siyaset","Kültür-Sanat","Teknoloji"])
    if request.method == "POST":
        text = request.form["haber_metni"]
        data = [text]
        sonuc = clf.predict(data).astype(int)
        return render_template("predict.html", prediction = classes[sonuc])
    else:

        return redirect(url_for("index"))
    
if __name__ == "__main__":
    app.run(debug = True)