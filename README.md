## About the Project
This project uses data from Appen to develop a model for an API that classifies disaster-related messages.

## Summary 
The project follows three steps:  
    1. ETL process  
    2. Model training  
    3. API integration  

## Files
The repository contains three folders, one for each step:

- **Data**
  Contains the csv files with the data disaster_categories.csv and disaster_messages.csv. The file process_data.py with the ETL code.

- **Models**
  Contains the file train_classifier.py with the model code.

- **App**
  Contains the file run.py that runs the API integration.



## Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl` 

2. On the `app` directory run the web app: `python run.py` 
