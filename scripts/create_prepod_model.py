import pandas as pd
from neuralprophet import NeuralProphet, set_log_level
import pickle

set_log_level("ERROR")

df = pd.read_csv('https://raw.githubusercontent.com/drdevinhopkins/hourly-report/main/data/since-2020.csv')
df.ds = pd.to_datetime(df.ds)
df = df.drop(['Date', 'Time'], axis=1)
df = df.sort_values(by='ds', ascending=True)
df.rename(columns = {'Triage hallway pts':'y', 'Adm. requests cum':"Adm requests cum", 'Pts.waiting for admission CUM':'Pts waiting for admission CUM'}, inplace = True)
df.dropna(inplace=True)
regressors = df.columns.tolist()
regressors.remove('y')
regressors.remove('ds')

m = NeuralProphet(
  # growth='off',
  yearly_seasonality=False,
  weekly_seasonality=True,
  daily_seasonality=True,
  n_lags=48,
  n_forecasts=12,
  changepoints_range=0.95,
  n_changepoints=50,
  num_hidden_layers=4,
  d_hidden=36,
  learning_rate=0.005,
)
m = m.add_lagged_regressor(names=regressors)
m = m.add_country_holidays("CA")
metrics = m.fit(df, 
                freq='H', 
                # progress='plot'
                )
print(metrics.tail(1))

with open('models/prepod_forecast_model.pkl', "wb") as f:
    pickle.dump(m, f)