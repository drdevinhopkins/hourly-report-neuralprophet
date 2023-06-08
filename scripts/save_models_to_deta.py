import os
from deta import Deta
from dotenv import load_dotenv

load_dotenv()

deta = Deta(os.environ.get("DETA_PROJECT_KEY"))

# SAVE FORECASTS TO DETA DRIVE

models = deta.Drive("models")

for model in ['daily-visits', 'daily-visits-with-weather', 'inflow', 'inflow-with-weather', 'prepod', 'stretcher-occupancy', 'verticalTBS']:
    try:
        models.put(model+'.np', path='models/'+model+'.np')
    except:
        print('Unable to upload '+model)

# models.put('daily-visits.np', path='models/daily-visits.np')
# models.put('daily-visits-with-weather.np',
#            path='models/daily-visits-with-weather.np')
# models.put('inflow.np', path='models/inflow.np')
# models.put('inflow-with-weather.np',
#            path='models/inflow-with-weather.np')
# models.put('prepod.np', path='models/prepod.np')
# models.put('stretcher-occupancy.np',
#            path='models/stretcher-occupancy.np')
# models.put('verticalTBS.np', path='models/verticalTBS.np')
