from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import requests

app = Flask(__name__)

MODEL_PATH = "model_artifacts.pkl"


# ── Download Model (Streaming) ─────────────────────────────────────────
def download_model():
    url = "https://huggingface.co/Nobyz/train-delay-model/resolve/main/model_artifacts.pkl"
    print("Downloading model from Hugging Face...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Model downloaded successfully!")


# ── Load Model ─────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    download_model()

with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

model          = artifacts["model"]
scaler         = artifacts["scaler"]
label_encoders = artifacts["label_encoders"]
FEATURES       = artifacts["features"]


# ── Data Options ──────────────────────────────────────────────────────
EGYPTIAN_STATIONS = [
    "Ramses", "Alexandria", "Aswan", "Luxor", "Port Said",
    "Suez", "Mansoura", "Tanta", "Zagazig", "Ismailia",
    "Minya", "Asyut", "Sohag", "Qena", "Beni Suef",
    "Damanhur", "Kafr El Sheikh", "Shibin El Kom", "Nag Hammadi", "Edfu",
]

EGYPTIAN_TRAINS = [
    "1010","1902","3006","934","3502","1","163","533","511","2007",
    "3015","119","377","593","535","941","974","321","903","1006",
    "945","379","7","537","978","1205","1131","965","539","1038",
    "1113","936","513","980","2025","901","381","1015","1109","1089",
    "3008","142","80","905","185","543","967","1004","911","2010",
    "1211","383","158","545","89","951","333","913","982","547",
    "15","160","385","998","949","549","1203","186","986","2023",
    "341","523","3017","162","389","919","121","955","325","551",
    "917","915","1915","21","164","343","957","1110","1191","923",
    "990","553","807","972","23","391","835","3023","595","2006",
    "563","2012","393","961","872","188","921","525","809","557",
    "925","1014","2027","2030","988","3009","969","1086","395","196",
    "157","86","959","123","976","931","1012","31","339","2014",
    "82","17","88","561","29","963","996","90","397","1088",
    "935","2008","3007","35","529","1008","327","890"
]

WIND_VALUES = [
    "light winds", "gentle breeze", "moderate breeze", "fresh breeze",
    "strong breeze", "light winds from the N", "light winds from the S",
    "light winds from the E", "light winds from the W",
    "gentle breeze from the N", "gentle breeze from the S",
    "moderate breeze from the N", "moderate breeze from the S",
    "fresh breeze from the N", "fresh breeze from the W",
]

WEATHER_VALUES = [
    "sunny", "cloudy", "overcast", "haze", "heavy haze", "moderate haze",
    "fog", "dense fog", "light rain", "moderate rain", "heavy rain",
    "thundershowers", "showers", "light to moderate rain",
    "moderate to heavy rain", "downpour", "dust storm",
]


# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Egyptian Railway Delay Predictor API",
        "version": "1.0"
    })


@app.route("/stations", methods=["GET"])
def get_stations():
    return jsonify({"stations": EGYPTIAN_STATIONS})


@app.route("/trains", methods=["GET"])
def get_trains():
    return jsonify({"trains": EGYPTIAN_TRAINS})


@app.route("/options", methods=["GET"])
def get_options():
    return jsonify({
        "stations": EGYPTIAN_STATIONS,
        "trains": EGYPTIAN_TRAINS,
        "directions": ["up", "down"],
        "wind_values": WIND_VALUES,
        "weather_values": WEATHER_VALUES,
        "months": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required = [
            "train_number","train_direction","station_from","station_to",
            "departure_time","arrival_time","month","year","wind","weather"
        ]

        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        def time_str_to_minutes(t):
            h, m = map(int, t.split(":"))
            return h * 60 + m

        encoded = {}
        for col in ["train_number", "train_direction", "station_name", "wind", "weather"]:
            le  = label_encoders[col]
            val = data["station_from"] if col == "station_name" else data[col]
            encoded[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

        row = [
            encoded["train_number"],
            encoded["train_direction"],
            encoded["station_name"],
            encoded["wind"],
            encoded["weather"],
            time_str_to_minutes(data["departure_time"]),
            time_str_to_minutes(data["arrival_time"]),
            int(data["month"]),
            int(data["year"]),
        ]

        X        = np.array(row).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred     = model.predict(X_scaled)[0]
        delay    = round(float(pred), 1)

        # expected arrival
        arr_h, arr_m = map(int, data["arrival_time"].split(":"))
        total_minutes = arr_h * 60 + arr_m + round(delay)
        exp_h = (total_minutes // 60) % 24
        exp_m = total_minutes % 60

        return jsonify({
            "predicted_delay_minutes": delay,
            "scheduled_arrival": data["arrival_time"],
            "expected_arrival": f"{str(exp_h).zfill(2)}:{str(exp_m).zfill(2)}",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)