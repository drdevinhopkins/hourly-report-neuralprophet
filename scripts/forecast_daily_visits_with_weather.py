import pandas as pd
from neuralprophet import NeuralProphet, set_log_level
import pickle

set_log_level("ERROR")

loaded_model = pickle.load(
    open('models/daily_visits_forecast_model_with_weather.pkl', 'rb'))

df = pd.read_csv(
    'https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/daily-visits.csv')
df.ds = pd.to_datetime(df.ds)
df = df.sort_values(by='ds')
df = df.reset_index(drop=True)

last_timestamp = df.loc[len(df)-1].ds

weather = pd.read_csv(
    'https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/daily-weather.csv')
weather = weather.rename(columns={'datetime': 'ds'})
weather.ds = pd.to_datetime(weather.ds)
weather = weather[['ds', 'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipcover', 'snow',
                   'snowdepth', 'windgust', 'windspeed', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'moonphase']]

df = df.merge(weather, on='ds')

future = pd.concat(
    [df.tail(4*7+1), weather[weather.ds > last_timestamp].head(8)])

forecast = loaded_model.predict(future, raw=True, decompose=False)

start = forecast.values.tolist()[0][0]
forecast_length = len(forecast.values.tolist()[0][1:])

forecast_output = pd.DataFrame(columns=['ds', 'visits', 'timestamp'])
forecast_output['ds'] = pd.date_range(
    start=start, periods=forecast_length, freq='D')
forecast_output['visits'] = forecast.values.tolist()[0][1:]
forecast_output['timestamp'] = pd.Timestamp.now().round(
    'S').replace(tzinfo=None)
forecast_output.to_csv('forecasts/daily_visits_with_weather.csv', index=False)
print(forecast_output)
