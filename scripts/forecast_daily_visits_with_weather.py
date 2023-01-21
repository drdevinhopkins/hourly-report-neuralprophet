import pandas as pd
from neuralprophet import NeuralProphet, set_log_level, load
import comet_ml
import os

comet_ml_api_key = os.environ.get("COMET_ML_API_KEY")
api = comet_ml.api.API(comet_ml_api_key)

api.download_registry_model("drdevinhopkins", "daily-visits-with-weather",
                            version="1.0.0", output_path="./comet-ml-models/", expand=True, stage=None)

set_log_level("ERROR")

# loaded_model = pickle.load(
#     open('models/daily_visits_forecast_model_with_weather.pkl', 'rb'))
loaded_model = load("comet-ml-models/daily-visits-with-weather.np")

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
# df['ID'] = 'test'

future = pd.concat(
    [df.tail(4*7+1), weather[weather.ds > last_timestamp].head(8)])
# future['ID'] = 'test'

# forecast = loaded_model.predict(future, raw=True, decompose=False)

# start = forecast.values.tolist()[0][0]
# forecast_length = len(forecast.values.tolist()[0][1:])

# forecast_output = pd.DataFrame(columns=['ds', 'visits', 'timestamp'])
# forecast_output['ds'] = pd.date_range(
#     start=start, periods=forecast_length, freq='D')
# forecast_output['visits'] = forecast.values.tolist()[0][1:]
# forecast_output['timestamp'] = pd.Timestamp.now().round(
#     'S').replace(tzinfo=None)
# forecast_output.to_csv('forecasts/daily_visits_with_weather.csv', index=False)
# print(forecast_output)


forecast = loaded_model.predict(future,
                                # raw=False,
                                # decompose=False
                                )

forecast_trimmed = forecast[forecast.y.isnull()].set_index('ds')
forecast_trimmed.to_csv('forecasts/daily_visits_with_weather_raw.csv')

start = forecast_trimmed.index.tolist()[0]
forecast_length = len(forecast_trimmed.index.tolist())
i = 0
timestamp = pd.Timestamp.now().round('S').replace(tzinfo=None)
forecast_output_list = []
for ds in pd.date_range(start=start, periods=forecast_length, freq='D'):
    i = i+1
    forecast_output_list.append(
        {'ds': ds, 'visits': forecast_trimmed.loc[ds]['yhat'+str(i)], 'timestamp': timestamp})
forecast_output = pd.DataFrame(forecast_output_list)
forecast_output.to_csv('forecasts/daily_visits_with_weather.csv', index=False)
