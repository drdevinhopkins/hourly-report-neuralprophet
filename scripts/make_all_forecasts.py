import pandas as pd
from neuralprophet import NeuralProphet, set_log_level, load
import comet_ml
import os
from deta import Deta
from dotenv import load_dotenv
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

load_dotenv()
set_log_level("ERROR")

deta = Deta(os.environ.get("DETA_PROJECT_KEY"))
comet_ml_api_key = os.environ.get("COMET_ML_API_KEY")
api = comet_ml.api.API(comet_ml_api_key)


# DAILY VISITS
try:
    api.download_registry_model("drdevinhopkins", "daily-visits",
                                version="1.0.0", output_path="comet-ml-models", expand=True, stage=None)


    loaded_model = load("comet-ml-models/daily-visits.np")

    df = pd.read_csv(
        'data/daily-visits.csv')
    df.ds = pd.to_datetime(df.ds)
    df = df.sort_values(by='ds')

    future = loaded_model.make_future_dataframe(df, periods=8)
    forecast = loaded_model.predict(future)

    forecast_trimmed = forecast[forecast.y.isnull()].set_index('ds')
    forecast_trimmed.to_csv('forecasts/daily_visits_raw.csv')

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
    forecast_output.to_csv('forecasts/daily_visits.csv', index=False)
except:
    print('Daily Visits forecast failed')


# DAILY VISITS WITH WEATHER
try:
    api.download_registry_model("drdevinhopkins", "daily-visits-with-weather",
                                version="1.0.0", output_path="comet-ml-models", expand=True, stage=None)

    loaded_model = load("comet-ml-models/daily-visits-with-weather.np")

    df = pd.read_csv(
        'daily-visits.csv')
    df.ds = pd.to_datetime(df.ds)
    df = df.sort_values(by='ds')
    df = df.reset_index(drop=True)

    # last_timestamp = df.loc[len(df)-1].ds

    weather = pd.read_csv(
        'data/daily-weather.csv')
    weather = weather.rename(columns={'datetime': 'ds'})
    weather.ds = pd.to_datetime(weather.ds)
    weather = weather[['ds', 'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipcover', 'snow',
                    'snowdepth', 'windgust', 'windspeed', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'moonphase']]

    df = df.merge(weather, how='right', on='ds', suffixes='')


    # future = pd.concat(
    #     [df.tail(4*7+1), weather[weather.ds > last_timestamp].head(8)])


    forecast = loaded_model.predict(df)

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
except:
    print('Daily visits with weather forecast failed')

# INFLOW
try:
    api.download_registry_model("drdevinhopkins", "inflow",
                                version="1.0.0", output_path="comet-ml-models", expand=True, stage=None)

    loaded_model = load("comet-ml-models/inflow.np")

    set_log_level("ERROR")

    target_column = 'Total Inflow hrly'
    df = pd.read_csv(
        'data/since-2020.csv')
    df.ds = pd.to_datetime(df.ds)
    df = df.drop(['Date', 'Time'], axis=1)
    df = df.sort_values(by='ds', ascending=True)
    df.rename(columns={target_column: 'y', 'Adm. requests cum': "Adm requests cum",
            'Pts.waiting for admission CUM': 'Pts waiting for admission CUM'}, inplace=True)
    df.dropna(inplace=True)
    regressors = df.columns.tolist()
    regressors.remove('y')
    regressors.remove('ds')

    future = loaded_model.make_future_dataframe(df, periods=12)
    forecast = loaded_model.predict(future)

    forecast_trimmed = forecast[forecast.y.isnull()].set_index('ds')
    forecast_trimmed.to_csv('forecasts/inflow_raw.csv')
    start = forecast_trimmed.index.tolist()[0]
    forecast_length = len(forecast_trimmed.index.tolist())
    i = 0
    timestamp = pd.Timestamp.now().round('S').replace(tzinfo=None)
    forecast_output_list = []
    for ds in pd.date_range(start=start, periods=forecast_length, freq='H'):
        i = i+1
        forecast_output_list.append(
            {'ds': ds, 'inflow': forecast_trimmed.loc[ds]['yhat'+str(i)], 'timestamp': timestamp})
    forecast_output = pd.DataFrame(forecast_output_list)
    forecast_output.to_csv('forecasts/inflow.csv', index=False)
except:
    print('Inflow forecast failed')

# INFLOW WITH WEATHER
try:
    api.download_registry_model("drdevinhopkins", "inflow-with-weather",
                                version="1.0.0", output_path="comet-ml-models", expand=True, stage=None)

    loaded_model = load("comet-ml-models/inflow-with-weather.np")

    set_log_level("ERROR")

    target_column = 'Total Inflow hrly'
    df = pd.read_csv(
        'data/since-2020.csv')
    df.ds = pd.to_datetime(df.ds)
    df = df.drop(['Date', 'Time'], axis=1)
    df = df.sort_values(by='ds', ascending=True)
    df.rename(columns={target_column: 'y', 'Adm. requests cum': "Adm requests cum",
            'Pts.waiting for admission CUM': 'Pts waiting for admission CUM'}, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    regressors = df.columns.tolist()
    regressors.remove('y')
    regressors.remove('ds')

    weather = pd.read_csv(
        'data/weatherArchiveAndForecast.csv')
    weather.ds = pd.to_datetime(weather.ds)
    weather = weather.sort_values(by='ds', ascending=True)

    # last_timestamp = df.loc[len(df)-1].ds

    df = df.merge(weather, how='right', on='ds', suffixes='')


    # future = pd.concat(
    #     [df.tail(48), weather[weather.ds > last_timestamp].head(12)])


    forecast = loaded_model.predict(df)

    forecast_trimmed = forecast[forecast.y.isnull()].set_index('ds')
    forecast_trimmed.to_csv('forecasts/inflow_with_weather_raw.csv')

    start = forecast_trimmed.index.tolist()[0]
    forecast_length = len(forecast_trimmed.index.tolist())
    i = 0
    timestamp = pd.Timestamp.now().round('S').replace(tzinfo=None)
    forecast_output_list = []
    for ds in pd.date_range(start=start, periods=forecast_length, freq='H'):
        i = i+1
        forecast_output_list.append(
            {'ds': ds, 'inflow': forecast_trimmed.loc[ds]['yhat'+str(i)], 'timestamp': timestamp})
    forecast_output = pd.DataFrame(forecast_output_list)
    forecast_output.to_csv('forecasts/inflow_with_weather.csv', index=False)
except:
    print('Inflow with weather forecast failed')

# PREPOD
try:
    api.download_registry_model("drdevinhopkins", "prepod",
                                version="1.0.0", output_path="comet-ml-models", expand=True, stage=None)

    loaded_model = load("comet-ml-models/prepod.np")

    set_log_level("ERROR")

    df = pd.read_csv(
        'data/since-2020.csv')
    df.ds = pd.to_datetime(df.ds)
    df = df.drop(['Date', 'Time'], axis=1)
    df = df.sort_values(by='ds', ascending=True)
    df.rename(columns={'Triage hallway pts': 'y', 'Adm. requests cum': "Adm requests cum",
            'Pts.waiting for admission CUM': 'Pts waiting for admission CUM'}, inplace=True)
    df.dropna(inplace=True)
    regressors = df.columns.tolist()
    regressors.remove('y')
    regressors.remove('ds')

    future = loaded_model.make_future_dataframe(df, periods=12)

    forecast = loaded_model.predict(future
                                    )

    forecast_trimmed = forecast[forecast.y.isnull()].set_index('ds')
    forecast_trimmed.to_csv('forecasts/prepod_raw.csv')

    start = forecast_trimmed.index.tolist()[0]
    forecast_length = len(forecast_trimmed.index.tolist())
    i = 0
    timestamp = pd.Timestamp.now().round('S').replace(tzinfo=None)
    forecast_output_list = []
    for ds in pd.date_range(start=start, periods=forecast_length, freq='H'):
        i = i+1
        forecast_output_list.append(
            {'ds': ds, 'prepod': forecast_trimmed.loc[ds]['yhat'+str(i)], 'timestamp': timestamp})
    forecast_output = pd.DataFrame(forecast_output_list)
    forecast_output.to_csv('forecasts/prepod.csv', index=False)
except:
    print('Prepod forecast failed')


# STRETCHER OCCUPANCY
try:
    api.download_registry_model("drdevinhopkins", "stretcher-occupancy",
                                version="1.0.0", output_path="comet-ml-models", expand=True, stage=None)

    loaded_model = load("comet-ml-models/stretcher-occupancy.np")

    set_log_level("ERROR")

    target_column = 'Total Stretcher pts'
    df = pd.read_csv(
        'data/since-2020.csv')
    df.ds = pd.to_datetime(df.ds)
    df = df.drop(['Date', 'Time'], axis=1)
    df = df.sort_values(by='ds', ascending=True)
    df.rename(columns={target_column: 'y', 'Adm. requests cum': "Adm requests cum",
            'Pts.waiting for admission CUM': 'Pts waiting for admission CUM'}, inplace=True)
    df.dropna(inplace=True)
    regressors = df.columns.tolist()
    regressors.remove('y')
    regressors.remove('ds')

    future = loaded_model.make_future_dataframe(df, periods=12)


    forecast = loaded_model.predict(future)

    forecast_trimmed = forecast[forecast.y.isnull()].set_index('ds')
    forecast_trimmed.to_csv('forecasts/stretcher_occupancy_raw.csv')
    start = forecast_trimmed.index.tolist()[0]
    forecast_length = len(forecast_trimmed.index.tolist())
    i = 0
    timestamp = pd.Timestamp.now().round('S').replace(tzinfo=None)
    forecast_output_list = []
    for ds in pd.date_range(start=start, periods=forecast_length, freq='H'):
        i = i+1
        forecast_output_list.append(
            {'ds': ds, 'stretchers': forecast_trimmed.loc[ds]['yhat'+str(i)], 'timestamp': timestamp})
    forecast_output = pd.DataFrame(forecast_output_list)
    forecast_output.to_csv('forecasts/stretcher_occupancy.csv', index=False)
except:
    print('Stretcher occupancy forecast failed')

# VERTICAL TBS
try:
    api.download_registry_model("drdevinhopkins", "verticaltbs",
                                version="1.0.0", output_path="comet-ml-models", expand=True, stage=None)

    loaded_model = load("comet-ml-models/verticalTBS.np")

    set_log_level("ERROR")

    target_column = 'Total Vertical TBS'
    df = pd.read_csv(
        'data/since-2020.csv')
    df.ds = pd.to_datetime(df.ds)
    df = df.drop(['Date', 'Time'], axis=1)
    df = df.sort_values(by='ds', ascending=True)
    df.rename(columns={target_column: 'y', 'Adm. requests cum': "Adm requests cum",
            'Pts.waiting for admission CUM': 'Pts waiting for admission CUM'}, inplace=True)
    df.dropna(inplace=True)
    regressors = df.columns.tolist()
    regressors.remove('y')
    regressors.remove('ds')

    future = loaded_model.make_future_dataframe(df, periods=12)

    forecast = loaded_model.predict(future)

    forecast_trimmed = forecast[forecast.y.isnull()].set_index('ds')
    forecast_trimmed.to_csv('forecasts/verticalTBS_raw.csv')
    start = forecast_trimmed.index.tolist()[0]
    forecast_length = len(forecast_trimmed.index.tolist())
    i = 0
    timestamp = pd.Timestamp.now().round('S').replace(tzinfo=None)
    forecast_output_list = []
    for ds in pd.date_range(start=start, periods=forecast_length, freq='H'):
        i = i+1
        forecast_output_list.append(
            {'ds': ds, 'verticalTBS': forecast_trimmed.loc[ds]['yhat'+str(i)], 'timestamp': timestamp})
    forecast_output = pd.DataFrame(forecast_output_list)
    forecast_output.to_csv('forecasts/verticalTBS.csv', index=False)
except:
    print('VerticalTBS forecast failed')


# SAVE FORECASTS TO DETA DRIVE

forecasts = deta.Drive("forecasts")

for forecast in ['daily_visits', 'daily_visits_with_weather', 'inflow', 'inflow_with_weather_raw', 'prepod', 'stretcher_occupancy', 'verticalTBS']:
    try:
        forecasts.put(forecast+'.csv', path='forecasts/'+forecast+'.csv')
    except:
        print('Failed to upload to deta: '+forecast)

# forecasts.put('daily_visits.csv', path='forecasts/daily_visits.csv')
# forecasts.put('daily_visits_with_weather.csv',
#               path='forecasts/daily_visits_with_weather.csv')
# forecasts.put('inflow.csv', path='forecasts/inflow.csv')
# forecasts.put('inflow_with_weather_raw.csv',
#               path='forecasts/inflow_with_weather_raw.csv')
# forecasts.put('prepod.csv', path='forecasts/prepod.csv')
# forecasts.put('stretcher_occupancy.csv',
#               path='forecasts/stretcher_occupancy.csv')
# forecasts.put('verticalTBS.csv', path='forecasts/verticalTBS.csv')
