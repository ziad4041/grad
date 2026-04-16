from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

with open('model_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model          = artifacts['model']
scaler         = artifacts['scaler']
label_encoders = artifacts['label_encoders']
FEATURES       = artifacts['features']

EGYPTIAN_STATIONS = [
    'Ramses', 'Alexandria', 'Aswan', 'Luxor', 'Port Said',
    'Suez', 'Mansoura', 'Tanta', 'Zagazig', 'Ismailia',
    'Minya', 'Asyut', 'Sohag', 'Qena', 'Beni Suef',
    'Damanhur', 'Kafr El Sheikh', 'Shibin El Kom', 'Nag Hammadi', 'Edfu'
]

EGYPTIAN_TRAINS = [
    '1010','1902','3006','934','3502','1','163','533','511','2007',
    '3015','119','377','593','535','941','974','321','903','1006',
    '945','379','7','537','978','1205','1131','965','539','1038',
    '1113','936','513','980','2025','901','381','1015','1109','1089',
    '3008','142','80','905','185','543','967','1004','911','2010',
    '1211','383','158','545','89','951','333','913','982','547',
    '15','160','385','998','949','549','1203','186','986','2023',
    '341','523','3017','162','389','919','121','955','325','551',
    '917','915','1915','21','164','343','957','1110','1191','923',
    '990','553','807','972','23','391','835','3023','595','2006',
    '563','2012','393','961','872','188','921','525','809','557',
    '925','1014','2027','2030','988','3009','969','1086','395','196',
    '157','86','959','123','976','931','1012','31','339','2014',
    '82','17','88','561','29','963','996','90','397','1088',
    '935','2008','3007','35','529','1008','327','890'
]

WIND_VALUES = [
    'light winds', 'gentle breeze', 'moderate breeze', 'fresh breeze',
    'strong breeze', 'light winds from the N', 'light winds from the S',
    'light winds from the E', 'light winds from the W',
    'gentle breeze from the N', 'gentle breeze from the S',
    'moderate breeze from the N', 'moderate breeze from the S',
    'fresh breeze from the N', 'fresh breeze from the W',
]

WEATHER_VALUES = [
    'sunny', 'cloudy', 'overcast', 'haze', 'heavy haze', 'moderate haze',
    'fog', 'dense fog', 'light rain', 'moderate rain', 'heavy rain',
    'thundershowers', 'showers', 'light to moderate rain',
    'moderate to heavy rain', 'downpour', 'dust storm'
]

DIRECTIONS = ['up', 'down']

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Egyptian Railway Delay Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3d;
    --accent: #6c63ff;
    --accent2: #ff6584;
    --green: #00e5a0;
    --text: #e8e8f0;
    --muted: #6b6b8a;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'DM Mono', monospace;
    min-height: 100vh; overflow-x: hidden;
  }
  body::before {
    content: ''; position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(108,99,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(108,99,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
  }
  .container { position: relative; z-index: 1; max-width: 780px; margin: 0 auto; padding: 60px 24px 80px; }
  .header { text-align: center; margin-bottom: 56px; }
  .badge {
    display: inline-block; background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.3); color: var(--accent);
    font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
    padding: 6px 16px; border-radius: 20px; margin-bottom: 20px;
  }
  h1 {
    font-family: 'Syne', sans-serif; font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800; line-height: 1.1; margin-bottom: 14px;
    background: linear-gradient(135deg, #fff 0%, var(--accent) 60%, var(--accent2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .subtitle { color: var(--muted); font-size: 13px; letter-spacing: 0.5px; }
  .train-track { width: 100%; height: 2px; background: var(--border); margin: 32px 0; position: relative; overflow: hidden; }
  .train-track::after { content: '🚂'; position: absolute; top: -12px; font-size: 20px; animation: train 4s linear infinite; }
  @keyframes train { from { left: -40px; } to { left: calc(100% + 10px); } }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 40px; position: relative; overflow: hidden; }
  .card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--accent), transparent); }
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .form-group { display: flex; flex-direction: column; gap: 8px; }
  .form-group.full { grid-column: 1 / -1; }
  label { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); }
  select, input {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
    font-family: 'DM Mono', monospace; font-size: 13px; padding: 12px 16px;
    border-radius: 10px; outline: none; transition: border-color 0.2s, box-shadow 0.2s;
    width: 100%; appearance: none;
  }
  select:focus, input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(108,99,255,0.1); }
  select {
    cursor: pointer;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%236b6b8a' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 14px center; padding-right: 36px;
  }
  .divider { height: 1px; background: var(--border); margin: 28px 0; grid-column: 1 / -1; }
  .section-label {
    grid-column: 1 / -1; font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
    color: var(--accent); display: flex; align-items: center; gap: 10px;
  }
  .section-label::after { content: ''; flex: 1; height: 1px; background: rgba(108,99,255,0.2); }
  .btn {
    grid-column: 1 / -1; background: linear-gradient(135deg, var(--accent), #8b5cf6);
    color: #fff; border: none; padding: 16px; border-radius: 12px;
    font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; letter-spacing: 1px;
    cursor: pointer; transition: opacity 0.2s, transform 0.1s; margin-top: 8px;
  }
  .btn:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .result { margin-top: 28px; padding: 28px 32px; border-radius: 16px; display: none; animation: slideUp 0.4s ease; }
  @keyframes slideUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
  .result.show { display: block; }
  .result.good  { background: rgba(0,229,160,0.07); border: 1px solid rgba(0,229,160,0.25); }
  .result.medium { background: rgba(255,193,7,0.07); border: 1px solid rgba(255,193,7,0.25); }
  .result.bad   { background: rgba(255,101,132,0.07); border: 1px solid rgba(255,101,132,0.25); }
  .result-label { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
  .result-value { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; line-height: 1; }
  .result.good  .result-value { color: var(--green); }
  .result.medium .result-value { color: #ffc107; }
  .result.bad   .result-value { color: var(--accent2); }
  .result-details { margin-top: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .result-detail { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 14px 16px; }
  .result-detail-label { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
  .result-detail-value { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; }
  .result-status { margin-top: 14px; font-size: 13px; color: var(--muted); }
  .loader { display: none; text-align: center; padding: 20px; color: var(--muted); font-size: 13px; animation: pulse 1.2s ease infinite; }
  @keyframes pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }
  @media (max-width: 560px) { .form-grid { grid-template-columns: 1fr; } .card { padding: 28px 20px; } .result-value { font-size: 2.2rem; } .result-details { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="badge">Egyptian National Railways</div>
    <h1>Train Delay<br>Predictor</h1>
    <p class="subtitle">Predict your journey arrival time before the train moves</p>
  </div>
  <div class="train-track"></div>
  <div class="card">
    <form id="form">
      <div class="form-grid">

        <div class="section-label">Train Info</div>

        <div class="form-group">
          <label>Train Number</label>
          <select name="train_number" required>
            {% for v in trains %}<option value="{{ v }}">{{ v }}</option>{% endfor %}
          </select>
        </div>

        <div class="form-group">
          <label>Direction</label>
          <select name="train_direction" required>
            <option value="up">Up (North)</option>
            <option value="down">Down (South)</option>
          </select>
        </div>

        <div class="divider"></div>
        <div class="section-label">Journey</div>

        <div class="form-group">
          <label>From Station</label>
          <select name="station_from" required>
            {% for v in stations %}<option value="{{ v }}">{{ v }}</option>{% endfor %}
          </select>
        </div>

        <div class="form-group">
          <label>To Station</label>
          <select name="station_to" required>
            {% for v in stations %}<option value="{{ v }}">{{ v }}</option>{% endfor %}
          </select>
        </div>

        <div class="form-group">
          <label>Departure Time</label>
          <input type="time" name="departure_time" value="08:00" required>
        </div>

        <div class="form-group">
          <label>Scheduled Arrival</label>
          <input type="time" name="arrival_time" value="12:00" required>
        </div>

        <div class="form-group">
          <label>Month</label>
          <select name="month" required>
            {% for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 1) %}
            <option value="{{ i }}">{{ m }}</option>{% endfor %}
          </select>
        </div>

        <div class="form-group">
          <label>Year</label>
          <input type="number" name="year" value="2024" min="2020" max="2030" required>
        </div>

        <div class="divider"></div>
        <div class="section-label">Weather Conditions</div>

        <div class="form-group">
          <label>Wind</label>
          <select name="wind" required>
            {% for v in winds %}<option value="{{ v }}">{{ v }}</option>{% endfor %}
          </select>
        </div>

        <div class="form-group">
          <label>Weather</label>
          <select name="weather" required>
            {% for v in weathers %}<option value="{{ v }}">{{ v }}</option>{% endfor %}
          </select>
        </div>

        <button type="submit" class="btn">PREDICT MY JOURNEY</button>
      </div>
    </form>

    <div class="loader" id="loader">Analyzing your journey...</div>

    <div class="result" id="result">
      <div class="result-label">Expected Delay</div>
      <div class="result-value" id="result-delay">-</div>
      <div class="result-details">
        <div class="result-detail">
          <div class="result-detail-label">Scheduled Arrival</div>
          <div class="result-detail-value" id="scheduled-arrival">-</div>
        </div>
        <div class="result-detail">
          <div class="result-detail-label">Expected Arrival</div>
          <div class="result-detail-value" id="expected-arrival">-</div>
        </div>
      </div>
      <div class="result-status" id="result-status"></div>
    </div>
  </div>
</div>

<script>
function addMinutes(timeStr, mins) {
  const [h, m] = timeStr.split(':').map(Number);
  const total = h * 60 + m + Math.round(mins);
  const nh = Math.floor(total / 60) % 24;
  const nm = total % 60;
  return `${String(nh).padStart(2,'0')}:${String(nm).padStart(2,'0')}`;
}

document.getElementById('form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const loader = document.getElementById('loader');
  const result = document.getElementById('result');
  const btn    = document.querySelector('.btn');
  result.className = 'result';
  loader.style.display = 'block';
  btn.disabled = true;

  const data = Object.fromEntries(new FormData(this));

  try {
    const res  = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(data) });
    const json = await res.json();
    loader.style.display = 'none';
    btn.disabled = false;

    if (json.error) {
      document.getElementById('result-delay').textContent = 'Error';
      document.getElementById('result-status').textContent = json.error;
      result.className = 'result bad show';
      return;
    }

    const delay = json.predicted_delay_minutes;
    const scheduledArrival = data.arrival_time;
    const expectedArrival  = addMinutes(scheduledArrival, delay);

    document.getElementById('result-delay').textContent    = delay + ' min';
    document.getElementById('scheduled-arrival').textContent = scheduledArrival;
    document.getElementById('expected-arrival').textContent  = expectedArrival;

    let cls, status;
    if (delay <= 5)       { cls = 'good';   status = 'On time - minimal delay expected'; }
    else if (delay <= 20) { cls = 'medium'; status = 'Moderate delay - plan accordingly'; }
    else                  { cls = 'bad';    status = 'Significant delay expected'; }

    document.getElementById('result-status').textContent = status;
    result.className = `result ${cls} show`;

  } catch(err) {
    loader.style.display = 'none';
    btn.disabled = false;
    document.getElementById('result-delay').textContent = 'Error';
    document.getElementById('result-status').textContent = 'Could not connect to server';
    result.className = 'result bad show';
  }
});
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML,
        trains=EGYPTIAN_TRAINS, stations=EGYPTIAN_STATIONS,
        directions=DIRECTIONS, winds=WIND_VALUES, weathers=WEATHER_VALUES,
        enumerate=enumerate)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        def time_str_to_minutes(t):
            h, m = map(int, t.split(':'))
            return h * 60 + m

        encoded = {}
        for col in ['train_number', 'train_direction', 'station_name', 'wind', 'weather']:
            le  = label_encoders[col]
            # station_name uses station_from
            val = data['station_from'] if col == 'station_name' else data[col]
            encoded[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

        row = [
            encoded['train_number'],
            encoded['train_direction'],
            encoded['station_name'],
            encoded['wind'],
            encoded['weather'],
            time_str_to_minutes(data['departure_time']),
            time_str_to_minutes(data['arrival_time']),
            int(data['month']),
            int(data['year']),
        ]

        X        = np.array(row).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred     = model.predict(X_scaled)[0]

        return jsonify({'predicted_delay_minutes': round(float(pred), 1)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Egyptian Railway Delay Predictor running at http://localhost:5000")
    app.run(debug=True, port=8080)
