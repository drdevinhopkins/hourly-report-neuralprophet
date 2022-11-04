# import pandas as pd
# from neuralprophet import NeuralProphet, set_log_level
# import pickle

# set_log_level("ERROR")

# #################

# loaded_model = pickle.load(
#     open('models/daily_visits_forecast_model_with_weather.pkl', 'rb'))

# df = pd.read_csv(
#     'https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/daily-visits.csv')
# df.ds = pd.to_datetime(df.ds)
# df = df.sort_values(by='ds')
# df = df.reset_index(drop=True)

# last_timestamp = df.loc[len(df)-1].ds

# weather = pd.read_csv(
#     'https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/daily-weather.csv')
# weather = weather.rename(columns={'datetime': 'ds'})
# weather.ds = pd.to_datetime(weather.ds)
# weather = weather[['ds', 'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipcover', 'snow',
#                    'snowdepth', 'windgust', 'windspeed', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'moonphase']]

# df = df.merge(weather, on='ds')

# future = pd.concat(
#     [df.tail(4*7+1), weather[weather.ds > last_timestamp].head(8)])

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

# print('Daily visits with weather - completed')

# ####################

# loaded_model = pickle.load(open('models/daily_visits_forecast_model.pkl', 'rb'))

# df = pd.read_csv('https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/daily-visits.csv')
# df.ds = pd.to_datetime(df.ds)
# df = df.sort_values(by='ds')

# future = loaded_model.make_future_dataframe(df, periods=8) # periods=m.n_forecasts, n_historic_predictions=False

# forecast = loaded_model.predict(future, raw=True, decompose=False)

# start = forecast.values.tolist()[-1][0]
# forecast_length = len(forecast.values.tolist()[-1][1:])

# forecast_output = pd.DataFrame(columns=['ds','visits','timestamp'])
# forecast_output['ds'] = pd.date_range(start=start, periods=forecast_length, freq='D')
# forecast_output['visits'] = forecast.values.tolist()[-1][1:]
# forecast_output['timestamp'] = pd.Timestamp.now().round('S').replace(tzinfo=None)
# forecast_output.to_csv('forecasts/daily_visits.csv', index=False)

# print('Daily visits - completed')

# #####################

# loaded_model = pickle.load(open('models/inflow_forecast_model_with_weather.pkl', 'rb'))

# target_column = 'Total Inflow hrly'
# df = pd.read_csv('https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/since-2020.csv')
# df.ds = pd.to_datetime(df.ds)
# df = df.drop(['Date', 'Time'], axis=1)
# df = df.sort_values(by='ds', ascending=True)
# df.rename(columns = {target_column:'y', 'Adm. requests cum':"Adm requests cum", 'Pts.waiting for admission CUM':'Pts waiting for admission CUM'}, inplace = True)
# df.dropna(inplace=True)
# df = df.reset_index(drop=True)
# regressors = df.columns.tolist()
# regressors.remove('y')
# regressors.remove('ds')

# weather = pd.read_csv('https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/weatherArchiveAndForecast.csv')
# weather.ds = pd.to_datetime(weather.ds)
# weather = weather.sort_values(by='ds', ascending=True)

# last_timestamp = df.loc[len(df)-1].ds

# df = df.merge(weather, on='ds')

# future = pd.concat([df.tail(48),weather[weather.ds>last_timestamp].head(12)])

# # future = loaded_model.make_future_dataframe(df, periods=12) # periods=m.n_forecasts, n_historic_predictions=False

# forecast = loaded_model.predict(future, raw=True, decompose=False)

# start = forecast.values.tolist()[0][0]
# forecast_length = len(forecast.values.tolist()[0][1:])

# forecast_output = pd.DataFrame(columns=['ds','inflow','timestamp'])
# forecast_output['ds'] = pd.date_range(start=start, periods=forecast_length, freq='H')
# forecast_output['inflow'] = forecast.values.tolist()[0][1:]
# forecast_output['timestamp'] = pd.Timestamp.now().round('S').replace(tzinfo=None)
# forecast_output.to_csv('forecasts/inflow_with_weather.csv', index=False)

# print('Inflow with weather - completed')

# ####################

# loaded_model = pickle.load(open('models/inflow_forecast_model.pkl', 'rb'))

# target_column = 'Total Inflow hrly'
# df = pd.read_csv('https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/since-2020.csv')
# df.ds = pd.to_datetime(df.ds)
# df = df.drop(['Date', 'Time'], axis=1)
# df = df.sort_values(by='ds', ascending=True)
# df.rename(columns = {target_column:'y', 'Adm. requests cum':"Adm requests cum", 'Pts.waiting for admission CUM':'Pts waiting for admission CUM'}, inplace = True)
# df.dropna(inplace=True)
# regressors = df.columns.tolist()
# regressors.remove('y')
# regressors.remove('ds')

# future = loaded_model.make_future_dataframe(df, periods=12) # periods=m.n_forecasts, n_historic_predictions=False

# forecast = loaded_model.predict(future, raw=True, decompose=False)

# start = forecast.values.tolist()[-1][0]
# forecast_length = len(forecast.values.tolist()[-1][1:])

# forecast_output = pd.DataFrame(columns=['ds','inflow','timestamp'])
# forecast_output['ds'] = pd.date_range(start=start, periods=forecast_length, freq='H')
# forecast_output['inflow'] = forecast.values.tolist()[-1][1:]
# forecast_output['timestamp'] = pd.Timestamp.now().round('S').replace(tzinfo=None)
# forecast_output.to_csv('forecasts/inflow.csv', index=False)

# print('Inflow - completed')

# ####################

# loaded_model = pickle.load(open('models/prepod_forecast_model.pkl', 'rb'))

# df = pd.read_csv('https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/since-2020.csv')
# df.ds = pd.to_datetime(df.ds)
# df = df.drop(['Date', 'Time'], axis=1)
# df = df.sort_values(by='ds', ascending=True)
# df.rename(columns = {'Triage hallway pts':'y', 'Adm. requests cum':"Adm requests cum", 'Pts.waiting for admission CUM':'Pts waiting for admission CUM'}, inplace = True)
# df.dropna(inplace=True)
# regressors = df.columns.tolist()
# regressors.remove('y')
# regressors.remove('ds')

# future = loaded_model.make_future_dataframe(df, periods=12) # periods=m.n_forecasts, n_historic_predictions=False

# forecast = loaded_model.predict(future, raw=True, decompose=False)

# start = forecast.values.tolist()[-1][0]
# forecast_length = len(forecast.values.tolist()[-1][1:])

# forecast_output = pd.DataFrame(columns=['ds','prepod','timestamp'])
# forecast_output['ds'] = pd.date_range(start=start, periods=forecast_length, freq='H')
# forecast_output['prepod'] = forecast.values.tolist()[-1][1:]
# forecast_output['timestamp'] = pd.Timestamp.now().round('S').replace(tzinfo=None)
# forecast_output.to_csv('forecasts/prepod.csv', index=False)

# print('Prepod - completed')

# ######################

# loaded_model = pickle.load(open('models/stretcher_occupancy_forecast_model.pkl', 'rb'))

# target_column = 'Total Stretcher pts'
# df = pd.read_csv('https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/since-2020.csv')
# df.ds = pd.to_datetime(df.ds)
# df = df.drop(['Date', 'Time'], axis=1)
# df = df.sort_values(by='ds', ascending=True)
# df.rename(columns = {target_column:'y', 'Adm. requests cum':"Adm requests cum", 'Pts.waiting for admission CUM':'Pts waiting for admission CUM'}, inplace = True)
# df.dropna(inplace=True)
# regressors = df.columns.tolist()
# regressors.remove('y')
# regressors.remove('ds')

# future = loaded_model.make_future_dataframe(df, periods=12) # periods=m.n_forecasts, n_historic_predictions=False

# forecast = loaded_model.predict(future, raw=True, decompose=False)

# start = forecast.values.tolist()[-1][0]
# forecast_length = len(forecast.values.tolist()[-1][1:])

# forecast_output = pd.DataFrame(columns=['ds','stretchers','timestamp'])
# forecast_output['ds'] = pd.date_range(start=start, periods=forecast_length, freq='H')
# forecast_output['stretchers'] = forecast.values.tolist()[-1][1:]
# forecast_output['timestamp'] = pd.Timestamp.now().round('S').replace(tzinfo=None)
# forecast_output.to_csv('forecasts/stretcher_occupancy.csv', index=False)

# print('Stretchers - completed')

# #########################

# loaded_model = pickle.load(open('models/verticalTBS_forecast_model.pkl', 'rb'))

# target_column = 'Total Vertical TBS'
# df = pd.read_csv('https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/since-2020.csv')
# df.ds = pd.to_datetime(df.ds)
# df = df.drop(['Date', 'Time'], axis=1)
# df = df.sort_values(by='ds', ascending=True)
# df.rename(columns = {target_column:'y', 'Adm. requests cum':"Adm requests cum", 'Pts.waiting for admission CUM':'Pts waiting for admission CUM'}, inplace = True)
# df.dropna(inplace=True)
# regressors = df.columns.tolist()
# regressors.remove('y')
# regressors.remove('ds')

# future = loaded_model.make_future_dataframe(df, periods=12) # periods=m.n_forecasts, n_historic_predictions=False

# forecast = loaded_model.predict(future, raw=True, decompose=False)

# start = forecast.values.tolist()[-1][0]
# forecast_length = len(forecast.values.tolist()[-1][1:])

# forecast_output = pd.DataFrame(columns=['ds','verticalTBS','timestamp'])
# forecast_output['ds'] = pd.date_range(start=start, periods=forecast_length, freq='H')
# forecast_output['verticalTBS'] = forecast.values.tolist()[-1][1:]
# forecast_output['timestamp'] = pd.Timestamp.now().round('S').replace(tzinfo=None)
# forecast_output.to_csv('forecasts/verticalTBS.csv', index=False)

# print('Vertical TBS - completed')