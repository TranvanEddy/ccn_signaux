#import des librairies
from flask import Flask, request, render_template
import numpy as np
import os
import io
import tensorflow as tf
import PIL
from PIL import Image
app = Flask(__name__)
model = tf.keras.models.load_model('best_model_signaux6.h5')
@app.route("/")
def home():
    return render_template("index.html")
#fonction prédictions
@app.route("/predict2", methods=["POST"])
def predict2():
    image = request.files["panneau"].read()
    image = open_img(np.array(image))
    result = predict_result(image)
    return render_template('index.html', zone_prediction="Le panneau est un : {}".format(liste[result]))

#fonction traitement de l'image de l'utilisateur
def open_img(img):
    to_predict = Image.open(io.BytesIO(img)).resize((28, 28))
    to_predict = np.array(to_predict).astype("float32")/255.0
    to_predict = to_predict.reshape(1, 28, 28, 3)
    return to_predict 

#fonction estimation
def predict_result(to_predict):
    prediction = model.predict(to_predict)
    proba_array = prediction.argsort()
    best_guess = proba_array[0][-1]
    return best_guess

#dictionnaire des classes
liste= {0:"double dos d'ane", 1:"simple dos d'ane", 2:"route glissante", 3:"virage à gauche", 4:"virage à droite", 
5:"route sinueuse", 6:"route sinueuse", 7:"traversée d'enfants", 8:"traversée de vélos", 9:"traversée de vaches", 
10:"attention travaux", 11:"feux tricolores", 12:"voie ferrée non sécurisée", 13:"attention", 14:"voie rétrécie", 
15:"voie rétrécie à gauche", 16:"voie rétrécie à droite", 17:"voie prioritaire à la prochaine intersection", 
18:"intersection", 19:"céder le passage", 20:"priorité au véhicule venant d'en face", 21:"stop", 22:"sens interdit", 
23:"interdit aux vélos", 24:"interdit au dessus d'un certain tonnage", 25:"interdit aux camions", 
26:"interdit au dessus d'une certaine largeur", 27:"interdit au dessus d'une certaine hauteur", 28:"interdit de circuler", 
29:"interdit de tourner à gauche", 30:"interdit de tourner à droite", 31:"interdiction de doubler", 32:"vitesse limitée", 
33:"voie réservée aux piétons et vélos", 34:"voie à sens unique", 35:"obligation e tourner", 
36:"obligation d'aller tout droit ou de tourner à droite", 37:"sens giratoire", 38:"voie cyclable", 
39:"voie cyclable et piétonnière", 40:"interdiction de stationner", 41:"interdiction de stationner et de s'arreter", 
42:"interdiction de stationner les 15 premiers jours du mois", 43:"interdictionde stationner les 15 derniers jours du mois", 
44:"priorité sur le véhicule venant d'en face", 45:"parking", 46:"parking handicapés", 47:"parking VL", 48:"parking PL", 
49:"parking bus", 50:"parking à l'essieu", 51:"jeux de ballons", 52:"interdit aux jeux de ballons", 53:"sens unique", 
54:"impasse", 55:"fin de travaux", 56:"passage pietons", 57:"passage de vélos", 58:"parking à droite", 
59:"route surélevée", 60:"fin de route prioritaire", 61:"route prioritaire"}

if __name__ == "__main__":
    app.run(debug=True)
