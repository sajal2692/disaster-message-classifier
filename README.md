# Disaster Message Classifier

This is a portfolio project to showcase Data Engineering skills including ETL and ML Pipeline preparation, utilising model in a web app, and data visualization. Offered by Udacity.

### Files:
- `data/process_data.py`: The ETL pipeline used to process data in preparation for model building.
- `models/train_classifier.py`: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle (pickle is not uploaded to the repo due to size constraints.).
- `app/templates/*.html`: HTML templates for the web app.
- `run.py`: Start the Python server for the web app and prepare visualizations.s

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

