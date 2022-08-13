import pickle
import pandas as pd
from neuralprophet import NeuralProphet, set_log_level

loaded_model = pickle.load(
    open('models/stretcher_occupancy_forecast_model.pkl', 'rb'))

target_column = 'Total Stretcher pts'
df = pd.read_csv(
    'https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/since-2020.csv')
df.ds = pd.to_datetime(df.ds)
df = df.drop(['Date', 'Time'], axis=1)
df = df.sort_values(by='ds', ascending=True)
df.rename(columns={target_column: 'y', 'Adm. requests cum': "Adm requests cum",
          'Pts.waiting for admission CUM': 'Pts waiting for admission CUM'}, inplace=True)
df.dropna(inplace=True)
regressors = df.columns.tolist()
regressors.remove('y')
regressors.remove('ds')
# df['ID'] = 'test'

# periods=m.n_forecasts, n_historic_predictions=False
future = loaded_model.make_future_dataframe(df, periods=12)

# forecast = loaded_model.predict(future, raw=True, decompose=False)

# start = forecast.values.tolist()[-1][0]
# forecast_length = len(forecast.values.tolist()[-1][1:])

# forecast_output = pd.DataFrame(columns=['ds','stretchers','timestamp'])
# forecast_output['ds'] = pd.date_range(start=start, periods=forecast_length, freq='H')
# forecast_output['stretchers'] = forecast.values.tolist()[-1][1:]
# forecast_output['timestamp'] = pd.Timestamp.now().round('S').replace(tzinfo=None)
# forecast_output.to_csv('forecasts/stretcher_occupancy.csv', index=False)


forecast = loaded_model.predict(future,
                                # raw=False,
                                # decompose=False
                                )

forecast_trimmed = forecast[forecast.y.isnull()].set_index('ds')
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
