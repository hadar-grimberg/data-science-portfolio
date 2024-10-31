# -*- coding: utf-8 -*-
"""
Hadar Grimberg
10/2/2021

"""

import requests
import pandas as pd
import numpy as np
import json


"""File to check whether the will-they-survive API is working"""
#reading test data
data=pd.read_csv('../data/raw/test.csv')
#converting it into dictionary
data=data.to_dict('records')
#packaging the data dictionary into a new dictionary
data_json={'data':data}

#defining the header info for the api request
headers = {
    'content-type': "application/json",
    'cache-control': "no-cache",
}

#making the api request
r=requests.get(url='http://localhost:8080//will-they-survive',headers=headers,data=json.dumps(data_json))

#getting the json data out
data=r.json()

#displaying the data
print(data)