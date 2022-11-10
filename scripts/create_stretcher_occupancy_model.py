from comet_ml import Experiment
from comet_ml.api import API, APIExperiment
import pandas as pd
from neuralprophet import NeuralProphet, set_log_level, save
import pickle
import os

experiment = Experiment(
    api_key=os.environ.get('COMET_ML_API_KEY'),
    project_name="stretcher-occupancy",
    workspace="drdevinhopkins",
)
api = API(os.environ.get('COMET_ML_API_KEY'))

set_log_level("ERROR")

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

params = {
    # growth='off',
    'yearly_seasonality': False,
    'weekly_seasonality': True,
    'daily_seasonality': True,
    'n_lags': 6,
    'n_forecasts': 24,
    'changepoints_range': 0.95,
    'n_changepoints': 50,
    'quantiles': [0.2, 0.5, 0.8]
    # num_hidden_layers=4,
    # d_hidden=36,
    # learning_rate=0.005,
}

m = NeuralProphet(**params)
m = m.add_lagged_regressor(names=regressors)
m = m.add_country_holidays("CA")
metrics = m.fit(df,
                freq='H',
                # progress='plot'
                )
print(metrics.tail(1))

experiment.log_parameters(params)
experiment.log_metrics(metrics.tail(1).iloc[0].to_dict())

save(m, "models/stretcher-occupancy.np")

experiment.log_model("stretcher-occupancy", "models/stretcher-occupancy.np")

with open('models/stretcher_occupancy_model.pkl', "wb") as f:
    pickle.dump(m, f)

experiment_name = experiment.name

experiment.end()


api.delete_registry_model('drdevinhopkins', 'stretcher-occupancy')

apiexp = api.get('drdevinhopkins/stretcher-occupancy/'+experiment_name)

apiexp.register_model('stretcher-occupancy')
