import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import random

random.seed(42)
np.random.seed(42)

# ── Egyptian Stations ─────────────────────────────────────────────────
stations = [
    'Ramses', 'Alexandria', 'Aswan', 'Luxor', 'Port Said',
    'Suez', 'Mansoura', 'Tanta', 'Zagazig', 'Ismailia',
    'Minya', 'Asyut', 'Sohag', 'Qena', 'Beni Suef',
    'Damanhur', 'Kafr El Sheikh', 'Shibin El Kom', 'Qena', 'Nag Hammadi'
]

# ── Train Numbers (Egyptian style) ───────────────────────────────────
train_numbers = [f'ENR-{i}' for i in range(100, 150)]

# ── Wind values (same as original) ───────────────────────────────────
wind_values = [
    'light winds from the S', 'light winds from the SE', 'light winds from the E',
    'light winds from the N', 'light winds from the NE', 'light winds from the NW',
    'gentle breeze from the N', 'gentle breeze from the NE', 'moderate breeze from the NE',
    'light winds from the SW', 'gentle breeze from the S', 'moderate breeze from the NW',
    'moderate breeze from the N', 'gentle breeze from the W', 'gentle breeze from the NW',
    'gentle breeze from the E', 'moderate breeze from the W', 'light winds',
    'gentle breeze from the SE', 'gentle breeze from the SW', 'moderate breeze from the E',
    'fresh breeze from the W', 'moderate breeze from the SE', 'fresh breeze from the NW',
    'fresh breeze from the N', 'moderate breeze from the S', 'fresh breeze from the NE',
    'gentle breeze', 'fresh breeze', 'moderate breeze', 'fresh breeze from the E',
    'strong breeze from the W', 'fresh breeze from the SE', 'moderate breeze from the SW',
    'strong breeze from the NW', 'strong breeze from the N', 'fresh breeze from the SW',
    'fresh breeze from the S', 'strong breeze from the SW', 'strong breeze from the NE',
    'moderate gale from the N'
]

# ── Weather (Egypt-appropriate — no snow/blizzard) ────────────────────
weather_values = [
    'sunny', 'light rain', 'cloudy', 'overcast', 'haze',
    'fog', 'moderate rain', 'thundershowers', 'showers',
    'light to moderate rain', 'moderate to heavy rain', 'heavy haze',
    'moderate haze', 'downpour', 'dense fog', 'dust storm'
]

# ── Generate 5000 rows ────────────────────────────────────────────────
n = 5000
rows = []

start_date = datetime(2023, 1, 1)
end_date   = datetime(2024, 12, 31)

egyptian_holidays = [
    '01-07', '01-25', '04-25', '05-01', '06-30', '07-23', '10-06', '11-04'
]

for i in range(n):
    # Random date
    rand_days = random.randint(0, (end_date - start_date).days)
    date = start_date + timedelta(days=rand_days)
    date_str = date.strftime('%Y-%m-%d')
    month_day = date.strftime('%m-%d')
    is_holiday = month_day in egyptian_holidays

    # Train info
    train_num  = random.choice(train_numbers)
    direction  = random.choice(['up', 'down'])
    station    = random.choice(stations)
    station_order = random.randint(1, 15)

    # Scheduled times
    sched_arr_h  = random.randint(0, 23)
    sched_arr_m  = random.choice([0, 15, 30, 45])
    sched_dep_m  = sched_arr_m + random.randint(2, 10)
    sched_dep_h  = sched_arr_h
    if sched_dep_m >= 60:
        sched_dep_m -= 60
        sched_dep_h = (sched_dep_h + 1) % 24

    sched_arr = time(sched_arr_h, sched_arr_m)
    sched_dep = time(sched_dep_h, sched_dep_m)

    # Stop time
    stop_time = random.choice([0, 2, 3, 5, 8, 10, '----'])

    # Delays (Egypt trains tend to have more delay 😅)
    base_delay = np.random.exponential(scale=8)
    if is_holiday:
        base_delay *= 1.5

    arrival_delay   = round(base_delay + np.random.normal(0, 2), 1)
    departure_delay = round(base_delay + np.random.normal(0, 2), 1)
    arrival_delay   = max(0, arrival_delay)
    departure_delay = max(0, departure_delay)

    # Actual times
    act_arr_min = sched_arr_h * 60 + sched_arr_m + arrival_delay
    act_dep_min = sched_dep_h * 60 + sched_dep_m + departure_delay
    act_arr_h, act_arr_m = int(act_arr_min // 60) % 24, int(act_arr_min % 60)
    act_dep_h, act_dep_m = int(act_dep_min // 60) % 24, int(act_dep_min % 60)

    actual_arr = time(act_arr_h, act_arr_m)
    actual_dep = time(act_dep_h, act_dep_m)

    # Weather
    wind    = random.choice(wind_values)
    weather = random.choice(weather_values)
    temp    = round(random.uniform(10, 42), 1)  # Egypt temperature range

    rows.append({
        'date':                     date_str,
        'train_number':             train_num,
        'train_direction':          direction,
        'station_name':             station,
        'station_order':            station_order,
        'scheduled_arrival_time':   sched_arr,
        'scheduled_departure_time': sched_dep,
        'stop_time':                stop_time,
        'actual_arrival_time':      actual_arr,
        'actual_departure_time':    actual_dep,
        'arrival_delay':            arrival_delay,
        'departure_delay':          departure_delay,
        'wind':                     wind,
        'weather':                  weather,
        'temperature':              temp,
        'major_holiday':            is_holiday,
    })

df = pd.DataFrame(rows)
df.to_csv('egyptian_test_data.csv', index=False)
print(f"✅ Done! Shape: {df.shape}")
print(df.head())
