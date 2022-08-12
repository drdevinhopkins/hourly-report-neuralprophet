import pandas as pd
from neuralprophet import NeuralProphet, set_log_level
import pickle

set_log_level("ERROR")

df = pd.read_csv(
    'https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/daily-visits.csv')
df.ds = pd.to_datetime(df.ds)
df = df.sort_values(by='ds')

weather = pd.read_csv(
    'https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/daily-weather.csv')
weather = weather.rename(columns={'datetime': 'ds'})
weather.ds = pd.to_datetime(weather.ds)
weather = weather[['ds', 'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipcover', 'snow',
                   'snowdepth', 'windgust', 'windspeed', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'moonphase']]

df = df.merge(weather, on='ds')

regressors = df.columns.tolist()
regressors.remove('y')
regressors.remove('ds')


m = NeuralProphet(
    # yearly_seasonality=False,
    # weekly_seasonality=True,
    # daily_seasonality=False,
    n_lags=4*7+1,
    n_forecasts=8,
    changepoints_range=0.95,
    n_changepoints=50,
    # num_hidden_layers=4,
    # d_hidden=36,
    # learning_rate=0.005,
).add_country_holidays("CA")

for reg in regressors:
    m = m.add_future_regressor(name=reg)

metrics = m.fit(df,
                freq='D',
                # progress='plot'
                )
print(metrics.tail(1))

with open('models/daily_visits_forecast_model_with_weather.pkl', "wb") as f:
    pickle.dump(m, f)
