import pandas as pd
from neuralprophet import NeuralProphet, set_log_level
import pickle

set_log_level("ERROR")

loaded_model = pickle.load(
    open('models/daily_visits_forecast_model.pkl', 'rb'))

df = pd.read_csv(
    'https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/daily-visits.csv')
df.ds = pd.to_datetime(df.ds)
df = df.sort_values(by='ds')
df['ID'] = 'test'

# periods=m.n_forecasts, n_historic_predictions=False
future = loaded_model.make_future_dataframe(df, periods=8)

# forecast = loaded_model.predict(future, raw=True, decompose=False)

# start = forecast.values.tolist()[-1][0]
# forecast_length = len(forecast.values.tolist()[-1][1:])

# forecast_output = pd.DataFrame(columns=['ds', 'visits', 'timestamp'])
# forecast_output['ds'] = pd.date_range(
#     start=start, periods=forecast_length, freq='D')
# forecast_output['visits'] = forecast.values.tolist()[-1][1:]
# forecast_output['timestamp'] = pd.Timestamp.now().round(
#     'S').replace(tzinfo=None)
# forecast_output.to_csv('forecasts/daily_visits.csv', index=False)


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
        {'ds': ds, 'visits': forecast_trimmed.loc[ds]['yhat'+str(i)], 'timestamp': timestamp})
forecast_output = pd.DataFrame(forecast_output_list)
forecast_output.to_csv('forecasts/daily_visits.csv', index=False)
