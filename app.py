from flask import Flask, request, render_template, jsonify
# Alternatively can use Django, FastAPI, or anything similar
from src.pipelines.pred_pipeline import CustomData, PredictPipeline
from src.utils import feature_engg

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')
@app.route('/predict', methods = ['POST', "GET"])

def predict_datapoint(): 
    if request.method == "GET": 
        return render_template("form.html")
    else: 
        data = CustomData(
            trans_date_trans_time = (request.form.get('cc_freq')),
            dob = (request.form.get('cc_freq_class')),
            lat= (request.form.get("distance_km")), 
            long = (request.form.get("month")),
            merch_lat = (request.form.get("day")), 
            merch_long = (request.form.get("hour")), 
            cc_num = (request.form.get("cc_num")), 
            amt = (request.form.get("amt")), 
            zip = request.form.get("zip"), 
            gender = request.form.get("gender"), 
            merchant = request.form.get("merchant"), 
            category = request.form.get("category"),
            city = request.form.get("city"),
            job = request.form.get("job")
        )
    new_data = data.get_data_as_dataframe()
    data = feature_engg(new_data)
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(data)

    if pred == 0:
        results = "You have made a Legitimate Transaction"
    else:
        results = "ALERT!!!!! \n You have made a Fraudulent Transaction"

    return render_template("results.html", final_result = results)

if __name__ == "__main__": 
    app.run(host = "0.0.0.0", debug= True)

#http://127.0.0.1:5000/ in browser