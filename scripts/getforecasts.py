from deta import Deta
import os
from dotenv import load_dotenv

load_dotenv()
deta = Deta(os.environ.get("DETA_PROJECT_KEY"))


drive = deta.Drive("forecasts")

for file in drive.list()['names']:
  myFile = drive.get(file)
  with open('forecasts/'+file, "wb+") as f:
    for chunk in myFile.iter_chunks(4096):
        f.write(chunk)
  myFile.close()
