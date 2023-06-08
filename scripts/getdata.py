from deta import Deta
import os

deta = Deta(os.environ.get("DETA_PROJECT_KEY"))

drive = deta.Drive("data")

for file in drive.list()['names']:
  myFile = drive.get(file)
  with open('data/'+file, "wb+") as f:
    for chunk in myFile.iter_chunks(4096):
        f.write(chunk)
  myFile.close()
