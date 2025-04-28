from flask import Flask, request, jsonify, render_template

from inference import fwi_prediction
import pickle

application = Flask(__name__)
app = application


ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
scaler_model = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("prediction.html")
    temperature = float(request.form.get("Temperature"))
    rh = float(request.form.get("RH"))
    ws = float(request.form.get("Ws"))
    rain = float(request.form.get("Rain"))
    ffmc = float(request.form.get("FFMC"))
    dmc = float(request.form.get("DMC"))
    isi = float(request.form.get("ISI"))
    classes = float(request.form.get("Classes"))
    region = float(request.form.get("Region"))
    result = fwi_prediction(temperature, rh, ws, rain, ffmc, dmc, isi, classes, region)
    # new_scaled_data = scaler_model.transform(
    #     [[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]]
    # )
    # result = ridge_model.predict(new_scaled_data)
    return render_template("prediction.html", results=result[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080", debug=True)
