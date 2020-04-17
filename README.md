
# DeepCOOH



# Introduction 

This project is about predicting COOH value with various features.

(So far I only considered (155) continuous features here 

and dropped all other features including some useful categorical features (7).



# Users Friendly Steps

1. !python train_imputer_and_model.py
    
    Save trained KNNImputer() as .pkl file.
    
    Save trained StandardScaler() as .pkl file.
    
    Save trained Sequential() model as .h5 file.

2. !python test_imputer_and_model.py
    
    Load trained KNNImputer() and impute test data
    
    Load trained StandardScaler() and standardize test data
    
    Load trained Sequential() model
    
    Make prediction and calculate R^2



# Files Descriptions

0. shinkong-cooh-infofab.json
    
    A configuration file that stored customized some parameter values.

1. train_imputer_and_model.py
    
    Impute and standardized data and train model with them.

2. test_imputer_and_model.py
    
    Evaluate the trained model performance with imputed and standardized test data.

3. _keras_model.py
    
    This is where we build the model and return the untrained model object.

4. _pandashandler.py
    
    Split train and test data or labeled and unlabeled data.

5. _plotsforml.py
    
    Make plots for checking the results or data itself.



# Author

Ethan Wang

yousenwang@gmail.com