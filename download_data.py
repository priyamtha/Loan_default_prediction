import gdown
import os

os.makedirs('data', exist_ok=True)

files = [
    {'id': '1Fg0Ez1i7qvrZMKpWjdzCA-oTkj3Gla0H', 'output': 'data/Loan_data.csv'},
    {'id': '1yT5RmPJ0rMmAttNC2bppVPL8Ks_fs6SI', 'output': 'data/Data_dictionary.csv'}
]

for f in files:
    url = f'https://drive.google.com/uc?id={f["id"]}'
    print(f'Downloading {f["output"]}...')
    gdown.download(url, f['output'], quiet=False)
