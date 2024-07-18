import comet_ml
import pandas as pd 

comet_ml.init(api_key="YOUR-COMET-API-KEY") 

# Fetch artifact containing your dataset
experiment = comet_ml.Experiment() 
housing_data_artifact = experiment.get_artifact('ckaiser/housing-data-baseline') 
housing_data_artifact.download('./datasets') 

# Load data into a Dataframe
housing_data = pd.read_csv('./datasets/housing-data.csv') 

# Import scikit libraries for training
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error 

# Initialize training parameters
params = { 
    "n_estimators": 1000, 
    "max_depth": 6, 
    "min_samples_split": 5, 
    "warm_start":True, 
    "oob_score":True, 
    "random_state": 42, 
}

# Split train and test datasets 
train, test = train_test_split( 
    housing_data, test_size=0.15, random_state=params['random_state'] 
) 

y_train = train['target'] 
x_train = train.drop(columns=['target']) 

y_test = test['target'] 
x_test = test.drop(columns=['target']) 

# Fit the model on the training data 
model = RandomForestRegressor(**params) 
model.fit(x_train, y_train) 

 
# Predict on the test set 
y_test_pred = model.predict(x_test) 

# Evaluate the model 
accuracy = mean_squared_error(y_test, y_test_pred) 
print(f'Validation Accuracy: {accuracy}') 

# Pickle and save model 
import pickle 
with open('./baseline.pkl', 'wb') as f: 
    pickle.dump(model, f) 

# Version and store model via Comet Artifacts 
model_artifact = comet_ml.Artifact('baseline-housing-model') 
model_artifact.add('./baseline.pkl') 
experiment.log_artifact(model_artifact) 