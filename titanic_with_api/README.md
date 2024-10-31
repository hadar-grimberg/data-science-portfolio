# algocart

## About the dataset:

- **Survived** is the predicting value. It is a binary variable where 1 is survived and 0 is not survived.  
- **PassengerID** and **Ticket** are assumed to be random unique identifiers, that have no impact on the outcome variable. Thus, they will be excluded from analysis.  
- **Pclass** is an ordinal variable of the ticket class, a proxy for socio-economic status (SES), representing 1 = upper class, 2 = middle class, and 3 = lower class.  
- **Name** is a nominal string variable, represemting the name of the passenger.  
- **Sex** is a categorical string variable, which indicating whether the passenger was female or male.  
- **Embarked** is a categorical string variable. it indicates the port of embarkation.  
- **Age** is a float continuous variable, the age of the passenger.  
- **Fare** is a float continuous variable, showing how much each passenger paid for their rather memorable journey.  
- **SibSp** is an integer discrete variable, which represents number of related siblings/spouse aboard.  
- **Parch** is an integer discrete variable, which represents number of related parents/children aboard.  
- **Cabin** is a nominal variable, representing the cabin number of each passenger.  

## About the program:
under the /code folder you can find the building the model pipeline.
- preprocessing.py file contains the preprocessing and the data engineering. there are many graphs and tables to explore the data. My conclusions are also included as comments
- model_selection.py file contains model selection and hyperparameter tuning.
- ensemble_model.py file take the best 5 models and combine them into ensemble in order to get better prediction.
- model.py file defines the model class.

under /data/raw you can find the raw data that used to build the model.
under /data/processed you can find the processed  data that used to feed the model.
under models you can find the pickle file of the trained model.

under the /api folder you can find code that relates to the api
- app.py is the code to create the api
- controller.py is a script that used to test the api

## How to
- First you need to clone the repository by `git clone https://github.com/hadar-grimberg/algocart.git`
- Second, install the requirements by `pip install -r /path/to/requirements.txt`
- Third, in the root directory of the repository, run `python api/app.py`

Now you can request for prediction, just send the passenger details to the server and you will get a prediction whether s.he survived or not.
The request pattern is:
```shell
curl -L -X GET "localhost:8080/will-they-survive" -H "content-type: application/json" --data-raw "{\"Age\": 40.7,\"Ticket\":\"h55666\",\"Pclass\":1, \"Sex\":\"female\", \"SibSp\":[2], \"Parch\":0, \"Fare\":40.5,\"Embarked\":\"C\"}"
```
Insertion Rules:
- Age & Fare - float
- Ticket - string
- Parch & SibSp - positive int (SibSp 0-8, Parch 0-6)
- Pclass - 1, 2 or 3
- Sex - "female" or "male"
- Embarked - 'C', 'Q' or 'S'

Enjoy =)
