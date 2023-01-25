from comet_ml import Experiment
from comet_ml.api import API, APIExperiment
import pandas as pd
from neuralprophet import NeuralProphet, set_log_level, save
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

experiment = Experiment(
    api_key=os.environ.get('COMET_ML_API_KEY'),
    project_name="daily-visits-with-weather",
    workspace="drdevinhopkins",
)
api = API(os.environ.get('COMET_ML_API_KEY'))

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

params = {
    # growth='off',
    # 'yearly_seasonality':False,
    # 'weekly_seasonality':True,
    # 'daily_seasonality':True,
    'n_lags': 4,
    'n_forecasts': 12,
    'changepoints_range': 0.95,
    'n_changepoints': 50,
    'quantiles': [0.2, 0.5, 0.8]
    # num_hidden_layers=4,
    # d_hidden=36,
    # learning_rate=0.005,
}

m = NeuralProphet(**params)
m = m.add_country_holidays("CA")

for reg in regressors:
    m = m.add_future_regressor(name=reg)

metrics = m.fit(df,
                freq='D',
                # progress='plot'
                )
print(metrics.tail(1))

experiment.log_parameters(params)
experiment.log_metrics(metrics.tail(1).iloc[0].to_dict())

save(m, "models/daily-visits-with-weather.np")

experiment.log_model("daily-visits-with-weather",
                     "models/daily-visits-with-weather.np")

# with open('models/daily_visits_forecast_model_with_weather.pkl', "wb") as f:
#     pickle.dump(m, f)

experiment_name = experiment.name

experiment.end()


api.delete_registry_model('drdevinhopkins', 'daily-visits-with-weather')

apiexp = api.get('drdevinhopkins/daily-visits-with-weather/'+experiment_name)

apiexp.register_model('daily-visits-with-weather')
